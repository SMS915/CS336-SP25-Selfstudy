import os
import re
import yaml
import argparse
import torch
import json
import wandb
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# 引入组件
from cs336_alignment.sft import (
    log_generations,
    tokenize_prompt_and_output,
    get_response_log_probs,
    masked_normalize
)
from cs336_alignment.grpo import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# ==========================================
# 1. 辅助函数：权重同步
# ==========================================
def sync_policy_to_vllm(policy_model: torch.nn.Module, vllm_instance: LLM):
    """
    将 PyTorch 训练模型的权重同步给 vLLM 推理引擎。
    这是单卡 GRPO 的核心生命线。
    """
    # 获取 vLLM 内部的模型引用
    vllm_model_executor = vllm_instance.llm_engine.model_executor
    
    # 注意：vLLM 0.7.x 架构可能有变，这里使用一种比较通用的方式
    # 如果是多卡，这里会很复杂。单卡相对简单。
    # 我们直接加载 state_dict
    
    # 1. 获取 PyTorch 模型权重 (CPU or GPU)
    state_dict = policy_model.state_dict()
    
    # 2. 推送给 vLLM
    # 这是一个 Hack 操作，但在单卡上有效
    # 也就是 Handout 提到的 load_policy_into_vllm_instance
    vllm_instance.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())

# ==========================================
# 2. 数据集 (只包含 Prompt)
# ==========================================
class GRPODataset(Dataset):
    def __init__(self, data_path, prompt_template=None, max_samples=None):
        self.prompts = []
        self.ground_truths = []
        with open(data_path, "r") as f:
            lines = f.readlines()
            if max_samples:
                lines = lines[:max_samples]
                
            for line in lines:
                item = json.loads(line)
                raw_prompt = item["prompt"] # SFT数据里的 prompt 字段
                if prompt_template:
                    p = prompt_template.replace("{question}", raw_prompt)
                    self.prompts.append(p)
                else:
                    self.prompts.append(raw_prompt)

                raw_response = item["response"]
                self.ground_truths.append(self._extract_answer(raw_response))
                    
        print(f"为GRPO训练加载了{len(self.prompts)}条样本.")

    def _extract_answer(self, text: str) -> str:
        """
        从完整的 SFT response 中提取纯答案部分。
        格式通常是: <think>...</think><answer>Content</answer>
        """
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return text.strip()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "ground_truth": self.ground_truths[idx]
        }

# ==========================================
# 3. 训练主循环
# ==========================================
def train(config_path: str):
    # --- Load Config ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    wandb.init(project=config["wandb"]["project"], name=config["wandb"]["run_name"], config=config)
    
    device = "cuda"
    output_dir = config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Tokenizer & Prompt ---
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    with open(config["data"]["prompt_path"], "r") as f:
        prompt_template = f.read()

    valid_examples = []
    print(f"加载{config['data']['valid_path']}条验证数据...")
    with open(config["data"]["valid_path"], "r") as f:
        for line in f:
            valid_examples.append(json.loads(line))
            
    val_prompts = []
    for ex in valid_examples:
        if prompt_template:
            formatted_prompt = prompt_template.replace("{question}", ex["problem"])
            val_prompts.append(formatted_prompt)
        else:
            val_prompts.append(ex["problem"])

    val_truths = [ex["solution"] for ex in valid_examples]

    # --- 1. Init Policy Model (PyTorch) ---
    print("加载策略模型 (Training)...")
    policy = AutoModelForCausalLM.from_pretrained(
        config["model"]["model_path"],
        torch_dtype=getattr(torch, config["model"]["dtype"]),
        attn_implementation=config["model"]["attn_implementation"],
        device_map="cuda", # 确保在 GPU 上
    )
    # 开启梯度检查点省显存
    policy.gradient_checkpointing_enable()
    policy.train()
    
    optimizer = AdamW(policy.parameters(), lr=float(config["training"]["learning_rate"]))

    # --- 2. Init vLLM (Generation) ---
    # 显存分配关键点：如果是 32G 显存，给 vLLM 40% (12.8G)，给 PyTorch 留 60%
    print("加载vllm (Generation)...")
    gpu_util = config["training"].get("gpu_memory_utlization", 0.4) # 默认改小
    
    llm = LLM(
        model=config["model"]["model_path"],
        dtype=config["model"]["dtype"],
        gpu_memory_utilization=gpu_util, 
        trust_remote_code=True,
        max_model_len=config["data"]["max_seq_length"], # 4096
        enforce_eager=True # 显存优化技巧
    )
    
    sampling_params = SamplingParams(
        temperature=config["training"]["sampling_temperature"],
        min_tokens=config["training"]["sampling_min_tokens"],
        max_tokens=config["training"]["sampling_max_tokens"],
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=config["training"]["group_size"] # 一次生成 G 个
    )

    # --- Data ---
    dataset = GRPODataset(
        config["data"]["train_path"], 
        prompt_template=prompt_template,
        max_samples=config["data"]["max_samples"]
    )
    # DataLoader 这里的 Batch Size 是 "多少个问题"
    # 实际生成的 Batch Size = rollout_batch_size / group_size
    questions_per_batch = config["training"]["rollout_batch_size"] // config["training"]["group_size"]
    
    dataloader = DataLoader(dataset, batch_size=questions_per_batch, shuffle=True, drop_last=True)
    
    # --- GRPO Loop ---
    n_grpo_steps = config["training"]["n_grpo_steps"]
    grad_accum_steps = config["training"]["gradient_accumulation_steps"]
    micro_batch_size = config["training"]["micro_batch_size"]
    epochs_per_batch = config["training"]["epochs_per_rollout_batch"]
    
    global_step = 0
    pbar = tqdm(total=n_grpo_steps, desc="GRPO Steps")
    
    # 无限循环数据，直到达到 n_grpo_steps
    data_iter = iter(dataloader)

    while global_step < n_grpo_steps:
        # -----------------------------------------
        # Phase 1: Experience Collection (Rollout)
        # -----------------------------------------
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        prompts = batch["prompt"]
        ground_truths = batch["ground_truth"] # List[str]
        
        # 1.1 同步权重: Policy -> vLLM
        # 这一步在单卡上必须做，确保 vLLM 用的是最新的参数生成
        sync_policy_to_vllm(policy, llm)
        
        # 1.2 生成 (vLLM)
        # outputs 是 list[RequestOutput]，每个包含 G 个 output
        generation_outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        
        # 1.3 整理数据
        all_prompts = []
        all_responses = []
        all_ground_truths = [] # 需要重复 G 次以匹配 response
        
        for i, req_output in enumerate(generation_outputs):
            q_prompts = [req_output.prompt] * config["training"]["group_size"]
            q_responses = [o.text for o in req_output.outputs]
            q_truths = [ground_truths[i]] * config["training"]["group_size"]
            
            all_prompts.extend(q_prompts)
            all_responses.extend(q_responses)
            all_ground_truths.extend(q_truths)
            
        # 1.4 计算奖励 (Reward & Advantage)
        # 这一步在 CPU 上做
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=all_responses,
            repeated_ground_truths=all_ground_truths,
            group_size=config["training"]["group_size"],
            advantage_eps=config["training"]["advantage_eps"],
            normalize_by_std=True
        )
        # 转为 Tensor 并移到 GPU
        advantages = torch.tensor(advantages).to(device).unsqueeze(1) # (B*G, 1)
        raw_rewards = torch.tensor(raw_rewards).to(device).unsqueeze(1)
        
        # 1.5 Tokenization (准备训练数据)
        # 复用 SFT 的 Tokenizer
        tokenized_batch = tokenize_prompt_and_output(
            prompt_strs=all_prompts,
            output_strs=all_responses,
            tokenizer=tokenizer,
            max_length=config["data"]["max_seq_length"]
        )
        
        input_ids = tokenized_batch["input_ids"].to(device)
        response_mask = tokenized_batch["response_mask"].to(device)
        labels = tokenized_batch["labels"].to(device)
        
        # 1.6 (Optional) Get Old Log Probs
        # 对于 Reinforce 其实不需要，但为了兼容 Clip Loss，这里通常会计算一次
        # 使用 no_grad
        with torch.no_grad():
            log_probs_dict = get_response_log_probs(policy, input_ids, labels)
            old_log_probs = log_probs_dict["log_probs"].detach() # (B*G, L)

        # -----------------------------------------
        # Phase 2: Optimization (Training)
        # -----------------------------------------
        # 创建一个临时的 Dataset/Loader 来进行 micro-batch 训练
        # 数据总量 = rollout_batch_size (例如 256)
        train_dataset_len = len(input_ids)
        indices = torch.randperm(train_dataset_len)
        
        policy.train()
        
        # 记录累积 loss
        step_loss = 0.0
        
        # Inner Epochs (On-policy 通常是 1)
        for _ in range(epochs_per_batch):
            for i in range(0, train_dataset_len, micro_batch_size):
                mb_idx = indices[i : i + micro_batch_size]
                
                # Micro-batch data
                mb_input_ids = input_ids[mb_idx]
                mb_labels = labels[mb_idx]
                mb_mask = response_mask[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                # mb_rewards = raw_rewards[mb_idx] # 如果是 no_baseline 需要这个
                
                # Forward
                mb_log_probs_dict = get_response_log_probs(policy, mb_input_ids, mb_labels)
                mb_policy_log_probs = mb_log_probs_dict["log_probs"]
                
                # GRPO Backward
                # 注意：grad_accum_steps 这里需要怎么算？
                # 我们希望每一轮 Rollout 更新一次参数。
                # 所以 accum_steps 应该是 (rollout_batch_size / micro_batch_size)
                actual_accum_steps = train_dataset_len // micro_batch_size
                
                loss, _ = grpo_microbatch_train_step(
                    policy_log_probs=mb_policy_log_probs,
                    response_mask=mb_mask,
                    gradient_accumulation_steps=actual_accum_steps,
                    loss_type=config["training"]["loss_type"],
                    advantages=mb_adv,
                    old_log_probs=mb_old_lp,
                    # raw_rewards=mb_rewards
                )
                
                step_loss += loss.item() / actual_accum_steps

            # End of Micro-batches -> Update
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config["training"]["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad()
        
        global_step += 1
        pbar.update(1)
        
        # --- Logging ---
        wandb.log({
            "train/loss": step_loss,
            "train/reward_mean": reward_meta["mean_reward"],
            "train/reward_max": reward_meta["max_reward"],
            "train/global_step": global_step
        })
        pbar.set_postfix(reward=reward_meta["mean_reward"])

        # --- Evaluation & Saving ---
        if global_step % config["evaluation"]["eval_every_steps"] == 0:
            # ... 调用 log_generations (和 SFT 一样) ...
            policy.eval()
            
            eval_max_tokens = config["evaluation"].get("max_new_tokens", 2048)
            eval_stats = log_generations(
                model=policy,
                tokenizer=tokenizer,
                prompts=val_prompts,
                ground_truths=val_truths,
                reward_fn=r1_zero_reward_fn,
                num_examples_to_log=config["evaluation"]["num_examples_to_log"],
                max_new_tokens=eval_max_tokens
            )
            
            # 合并日志
            eval_stats["train/global_step"] = global_step
            wandb.log(eval_stats, commit=False)
            
            policy.train() # 切回训练模式
            
        if global_step % config["training"]["save_steps"] == 0:
            print(f"保存checkpointing {global_step}")
            policy.save_pretrained(os.path.join(output_dir, f"step_{global_step}"))
            tokenizer.save_pretrained(os.path.join(output_dir, f"step_{global_step}"))

    print("训练完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo_config.yaml")
    args = parser.parse_args()
    train(args.config)