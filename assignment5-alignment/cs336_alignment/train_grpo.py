import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from cs336_alignment.bootstrap_runtime import bootstrap_cuda_visible_devices

bootstrap_cuda_visible_devices(default_config_path="configs/train/grpo_config.yaml")

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
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup 
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from vllm import LLM, SamplingParams
# 引入组件
from cs336_alignment.device_config import (
    apply_runtime_environment,
    build_model_load_kwargs,
    get_autocast_device_type,
    get_model_primary_device,
    get_vllm_load_kwargs,
    resolve_torch_device,
)
from cs336_alignment.sft import (
    log_generations_transformer,
    log_generations_vllm,
    get_response_log_probs,
)
from cs336_alignment.grpo import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step
)
from cs336_alignment.utils import tokenize_prompt_and_output, robust_reward_fn

# 权重同步辅助函数
def load_policy_into_vllm_instance(policy: torch.nn.Module, llm:LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
    del state_dict
    # torch.cuda.empty_cache()

# 数据集 (只包含 Prompt)
class GRPODataset(Dataset):
    def __init__(self, data_path, prompt_template=None, start_sample=None, max_samples=None):
        self.prompts = []
        self.ground_truths = []
        
        with open(data_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            if max_samples:
                start = start_sample if start_sample is not None else 0
                lines = lines[start : start + max_samples]
                
            for i, line in enumerate(lines):
                try:
                    item = json.loads(line)
                    
                    # 提取问题 (Problem / Question / Prompt)
                    raw_prompt = item.get("problem") or item.get("question") or item.get("prompt")
                    
                    # 提取答案, 优先取干净的 answer/solution，没有则从 response 提取
                    # 尝试获取直接答案
                    direct_answer = item.get("answer") or item.get("solution")
                    
                    if direct_answer is not None:
                        # 如果有直接答案，直接使用
                        final_answer = str(direct_answer).strip()
                    else:
                        # 如果没有直接答案，尝试从推理文本 response 中提取
                        raw_response = item.get("response")
                        if raw_response:
                            final_answer = self._extract_answer(str(raw_response))
                        else:
                            final_answer = None

                    if not raw_prompt or not final_answer:
                        print(f"第 {i} 行数据不完整，已跳过。")
                        continue

                    # 构造 Prompt
                    if prompt_template:
                        p = prompt_template.replace("{question}", str(raw_prompt).strip())
                        self.prompts.append(p)
                    else:
                        self.prompts.append(str(raw_prompt).strip())

                    self.ground_truths.append(final_answer)
                
                except Exception as e:
                    print(f"解析第 {i} 行出错: {e}")
                    continue
                    
        print(f"加载完成: {len(self.prompts)} 条样本用于 GRPO.")

    def _extract_answer(self, text: str) -> str:
        """从 SFT/RL 风格的文本中提取 <answer> 标签内的内容"""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "ground_truth": self.ground_truths[idx]
        }

# 训练主循环
def train(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    apply_runtime_environment(config)
    
    if config["model"]["wandb_id"] is None:
        wandb.init(project=config["wandb"]["project"], name=config["wandb"]["run_name"], config=config)
    else:
        wandb.init(
            project=config["wandb"]["project"],
            id=config["model"]["wandb_id"],   # 指定 ID
            resume="must",    # 强制续训，如果ID不存在会报错
            config=config
        )
    
    default_device = resolve_torch_device(config)
    dtype = getattr(torch, config["model"]["dtype"])
    output_dir = config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 加载分词器与prompt
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["model_path"],
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    with open(config["data"]["prompt_path"], "r") as f:
        prompt_template = f.read()

    valid_examples = []
    print(f"从加载{config['data']['valid_path']}验证数据...")
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

    # 初始化策略模型(PyTorch)
    print("加载策略模型 (Training)...")
    model_load_kwargs = build_model_load_kwargs(config)
    policy = AutoModelForCausalLM.from_pretrained(config["model"]["model_path"], **model_load_kwargs)
    if "device_map" not in model_load_kwargs:
        policy.to(default_device)
    device = get_model_primary_device(policy)
    device_type = get_autocast_device_type(device)
    print(f"训练设备入口: {device}")
    # 开启梯度检查点省显存
    if config["training"].get("gradient_checkpointing", True):
        policy.gradient_checkpointing_enable()
    policy.config.use_cache = False
    policy.train()

    
    optimizer = AdamW(policy.parameters(), lr=float(config["training"]["learning_rate"]))
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = float(config["training"]["learning_rate"])

    # 初始化vllm
    print("加载vllm (Generation)...")
    vllm_load_kwargs = get_vllm_load_kwargs(config, default_gpu_memory_utilization=0.4)
    if vllm_load_kwargs.get("tensor_parallel_size", 1) != 1:
        print("警告: 当前 RL 的权重同步逻辑主要按 tensor_parallel_size=1 验证，多卡 vLLM 需额外留意兼容性。")

    llm = LLM(model=config["model"]["model_path"], **vllm_load_kwargs)

    load_policy_into_vllm_instance(policy, llm)
    sampling_params = SamplingParams(
        temperature=config["training"]["sampling_temperature"],
        min_tokens=config["training"]["sampling_min_tokens"],
        max_tokens=config["training"]["sampling_max_tokens"],
        stop=config["training"].get("stop_tokens", ["</answer>", "<|endoftext|>"]),
        include_stop_str_in_output=True,
        n=config["training"]["group_size"], # 一次生成 G 个
        repetition_penalty=config["training"]["repetition_penalty"],
    )

    # --- Data ---
    dataset = GRPODataset(
        config["data"]["train_path"], 
        prompt_template=prompt_template,
        start_sample=config["data"]["start_sample"],
        max_samples=config["data"]["max_samples"]
    )
    # Batch Size = rollout_batch_size / group_size
    questions_per_batch = config["training"]["rollout_batch_size"] // config["training"]["group_size"]
    
    dataloader = DataLoader(dataset, batch_size=questions_per_batch, shuffle=False, drop_last=True)
    
    # 训练循环
    n_grpo_steps = config["training"]["n_grpo_steps"]
    micro_batch_size = config["training"]["micro_batch_size"]
    epochs_per_batch = config["training"]["epochs_per_rollout_batch"]
    clip_range = config["training"]["clip_range"]
    normalize_by_std = config["training"].get("normalize_by_std", True)
    remove_length_norm = config["training"].get("remove_length_norm", False)
    fixed_norm_length = config["training"].get("fixed_norm_length", 2048)
   
    start_step = config["model"]["start_step"]
    global_step = start_step if start_step is not None else 0
    if "warmup_ratio" in config["training"]:
        warmup_steps = int(n_grpo_steps * config["training"]["warmup_ratio"])

    print(f"Total Steps: {n_grpo_steps}, Warmup Steps: {warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=n_grpo_steps,
        last_epoch=global_step - 1 
    )

    print(f"从第{global_step}步继续训练")
    pbar = tqdm(total=n_grpo_steps, desc="GRPO Steps")
    pbar.update(global_step)
    # 无限循环数据，直到达到 n_grpo_steps
    data_iter = iter(dataloader)
    best_reward = config["model"].get("best_reward", 0.0)

    print(f"当前最佳reward为{best_reward}")
    while global_step < n_grpo_steps:
    # -----------------------------------------
    # 采样阶段
    # -----------------------------------------
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        prompts = batch["prompt"]
        ground_truths = batch["ground_truth"]
        
        # 生成 (vLLM)
        generation_outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        
        # 整理数据
        all_prompts = []
        all_responses = []
        all_ground_truths = [] # prompts 和 GT 需要重复 gp_size 次以匹配 response
        
        for i, req_output in enumerate(generation_outputs):
            q_prompts = [req_output.prompt] * config["training"]["group_size"]
            q_responses = [o.text for o in req_output.outputs]
            q_truths = [ground_truths[i]] * config["training"]["group_size"]
            
            all_prompts.extend(q_prompts)
            all_responses.extend(q_responses)
            all_ground_truths.extend(q_truths)
            
        # 计算奖励 (Reward & Advantage)
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=robust_reward_fn,
            rollout_responses=all_responses,
            repeated_ground_truths=all_ground_truths,
            group_size=config["training"]["group_size"],
            advantage_eps=config["training"]["advantage_eps"],
            normalize_by_std=normalize_by_std,
            length_panelty=config["training"].get("length_panelty", False)
        )
        # 转为 Tensor 并移到 GPU
        advantages = advantages.to(device).unsqueeze(1)
        raw_rewards = raw_rewards.to(device).unsqueeze(1)


        lengths = []
        
        for r in all_responses:
            lengths.append(len(tokenizer.encode(r))) # 估算 Token 长度
        
        # 准备训练数据
        tokenized_batch = tokenize_prompt_and_output(
            prompt_strs=all_prompts,
            output_strs=all_responses,
            tokenizer=tokenizer,
            max_length=config["data"]["max_seq_length"]
        )
        
        input_ids = tokenized_batch["input_ids"].to(device)
        response_mask = tokenized_batch["response_mask"].to(device)
        labels = tokenized_batch["labels"].to(device)
        attention_mask = tokenized_batch["attention_mask"].to(device)

        inference_batch_size = config["training"].get("inference_batch_size", 2)
        total_entropy = 0.0
        total_tokens = 0
        old_log_probs_list = []

        # 由于vllm和transformers的前向算子不一致，在这里用transformer库重新计算log_prob
        policy.eval()
        with torch.no_grad():
            with torch.amp.autocast(device_type=device_type, dtype=dtype):
                # 分批计算log_prob
                for i in tqdm(range(0, len(input_ids), inference_batch_size), desc="Calculating Ref LogProbs"):
                    batch_input_ids = input_ids[i : i + inference_batch_size]
                    batch_labels = labels[i : i + inference_batch_size]
                    batch_mask = attention_mask[i: i + inference_batch_size]

                    log_probs_dict = get_response_log_probs(policy,
                                                            batch_input_ids,
                                                            batch_mask,
                                                            batch_labels,
                                                            return_token_entropy=True)

                    if "token_entropy" in log_probs_dict:
                        entropy_sum = (log_probs_dict["token_entropy"] * batch_mask).sum().item()
                        token_count = batch_mask.sum().item()
                        total_entropy += entropy_sum
                        total_tokens += token_count
                    old_log_probs_list.append(log_probs_dict["log_probs"].detach().cpu())
                
        old_log_probs = torch.cat(old_log_probs_list, dim=0).to(device)
        policy.train() # 恢复 Train 模式

        mean_rollout_entropy = total_entropy / (total_tokens + 1e-8)
        wandb.log({"train/mean_token_entropy": mean_rollout_entropy}, commit=False)
        del total_entropy

    # -----------------------------------------
    # 优化阶段
    # -----------------------------------------
        train_dataset_len = len(input_ids)
        # 打乱
        indices = torch.randperm(train_dataset_len)
        
        policy.train()
        optimizer.zero_grad()
        # Inner Epochs (On-policy 通常是 1)
        for epoch_idx in range(epochs_per_batch):
            assert train_dataset_len % micro_batch_size == 0
            actual_accum_steps = train_dataset_len // micro_batch_size
            epoch_metrics = {"loss": [], "clip_ratio": [], "approx_kl": []}
            current_epoch_loss = 0.0
            current_epoch_clip = 0.0
            current_epoch_kl = 0.0

            for i in range(0, train_dataset_len, micro_batch_size):
                mb_idx = indices[i : i + micro_batch_size]
                
                # Micro-batch data
                mb_input_ids = input_ids[mb_idx]
                mb_labels = labels[mb_idx]
                mb_mask = response_mask[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                mb_attention_mask = attention_mask[mb_idx]
                # mb_rewards = raw_rewards[mb_idx] # 如果是 no_baseline 需要这个
                
                # Forward
                with torch.amp.autocast(device_type=device_type, dtype=dtype):
                    mb_log_probs_dict = get_response_log_probs(policy, mb_input_ids, mb_attention_mask, mb_labels)
                    mb_policy_log_probs = mb_log_probs_dict["log_probs"]

                    # GRPO Backward
                    loss, step_metrics = grpo_microbatch_train_step(
                        policy_log_probs=mb_policy_log_probs,
                        response_mask=mb_mask,
                        gradient_accumulation_steps=actual_accum_steps,
                        loss_type=config["training"]["loss_type"],
                        advantages=mb_adv,
                        old_log_probs=mb_old_lp,
                        cliprange=clip_range,
                        remove_length_norm=remove_length_norm,
                        fixed_norm_length = fixed_norm_length
                        # raw_rewards=mb_rewards
                    )

                current_epoch_loss += loss.item()
                current_epoch_clip += step_metrics["clip_ratio"].item() / actual_accum_steps
                current_epoch_kl += step_metrics["approx_kl"].item() / actual_accum_steps

            # End of Micro-batches -> Update
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config["training"]["max_grad_norm"])
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            epoch_metrics["loss"].append(current_epoch_loss)
            epoch_metrics["clip_ratio"].append(current_epoch_clip)
            epoch_metrics["approx_kl"].append(current_epoch_kl)
        current_lr = scheduler.get_last_lr()[0]
        global_step += 1
        pbar.update(1)
        
        wandb.log({
            # 核心表现
            "train/reward_mean": reward_meta["mean_reward"],
            "train/reward_std": raw_rewards.std().item(),     # 组间方差
            "train/format_rate": reward_meta["format_rate"],  # 格式正确率
            
            # 训练动态
            "train/loss": np.mean(epoch_metrics["loss"]),
            "train/clip_fraction": np.mean(epoch_metrics["clip_ratio"]),
            "train/approx_kl": np.mean(epoch_metrics["approx_kl"]),
            "train/lr": current_lr,
            "train/grad_norm": grad_norm.item(),
            
            # 行为特征
            "train/completion_len_mean": np.mean(lengths),
            "train/completion_len_max": np.max(lengths),
            
            # Step
            "train/global_step": global_step
        })
        # 同步权重到vllm，以确保on-policy RL时训练和推理模型的一致性
        load_policy_into_vllm_instance(policy, llm)
        
        pbar.set_postfix(reward=reward_meta["mean_reward"])

        if global_step % config["evaluation"]["eval_every_steps"] == 0:
            # policy.eval()
            # eval_stats = log_generations_transformer(
            #     model=policy,
            #     tokenizer=tokenizer,
            #     prompts=val_prompts,
            #     ground_truths=val_truths,
            #     reward_fn=robust_reward_fn,
            #     num_examples_to_log=config["evaluation"]["num_examples_to_log"],
            #     max_new_tokens=eval_max_tokens
            # )
            # policy.train()

            eval_stats = log_generations_vllm(
                llm=llm,  # 传入 vllm 实例
                prompts=val_prompts,
                ground_truths=val_truths,
                reward_fn=robust_reward_fn,
                num_examples_to_log=config["evaluation"]["num_examples_to_log"],
                max_new_tokens=config["evaluation"].get("max_new_tokens", 1024)
            )
            
            # 合并日志
            eval_stats["train/global_step"] = global_step
            wandb.log(eval_stats, commit=False)

            
        current_reward = reward_meta["mean_reward"]
        
        if current_reward > best_reward:
            best_reward = current_reward
            print(f"最佳reward: {global_step}时达到{best_reward:.4f}! 正在保存...")
            
            best_save_path = os.path.join(output_dir, "checkpoint-best")
            policy.save_pretrained(best_save_path)
            tokenizer.save_pretrained(best_save_path)
            
            # 在 WandB 里打个标记
            wandb.log({"train/best_reward": best_reward, "train/global_step": global_step}, commit=False)

        if global_step % config["training"]["save_steps"] == 0:
            print(f"定期保存中 {global_step}")
            
            # 保存路径带上步数
            step_save_path = os.path.join(output_dir, f"checkpoint-step-{global_step}")
            policy.save_pretrained(step_save_path)
            tokenizer.save_pretrained(step_save_path)


    print("训练完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/grpo_config.yaml")
    args = parser.parse_args()
    train(args.config)
