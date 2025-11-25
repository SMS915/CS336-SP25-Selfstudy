import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
# 2. 解决 CUDA 碎片化 (针对本次报错)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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

def robust_reward_fn(response: str, ground_truth: str) -> dict[str, float]:
    """
    包装官方的 reward_fn，增加对格式的鲁棒性处理。
    主要修复 </think><answer> 之间缺失空格的问题。
    """
    # 1. 修复空格问题
    cleaned_response = response.replace("</think><answer>", "</think> <answer>")
    
    # 2. (可选) 修复可能存在的换行问题
    cleaned_response = cleaned_response.replace("</think>\n<answer>", "</think> <answer>")
    
    # 3. 调用官方评分函数
    return r1_zero_reward_fn(cleaned_response, ground_truth)

# ==========================================
# 1. 辅助函数：权重同步
# ==========================================
def sync_policy_to_vllm(policy_model: torch.nn.Module, vllm_instance: LLM):
    """
    单卡专用：将 PyTorch 训练模型 (policy) 的权重原地更新到 vLLM 推理模型中。
    适配 vLLM 0.7.x 及 V0 Engine 架构 (engine_core).
    """
    # 1. 获取 LLMEngine
    llm_engine = vllm_instance.llm_engine
    vllm_model = None
    
    # 定义查找 Executor 的候选对象列表
    # 顺序：直接查找 -> 在 engine_core 里查找
    search_targets = [llm_engine]
    if hasattr(llm_engine, "engine_core"):
        search_targets.append(llm_engine.engine_core)
    
    executor = None
    
    # 2. 深度查找 Executor
    for target in search_targets:
        # 尝试常见的属性名
        if hasattr(target, "model_executor"):
            executor = target.model_executor
            break
        if hasattr(target, "executor"):
            executor = target.executor
            break
            
    if executor is None:
        # 调试信息：如果还是找不到，打印 engine_core 的属性
        print("❌ Error: 无法找到 executor。")
        if hasattr(llm_engine, "engine_core"):
            print(f"Debug: engine_core attrs: {[a for a in dir(llm_engine.engine_core) if not a.startswith('_')]}")
        raise RuntimeError("无法定位 vLLM Executor，请检查 vLLM 版本结构。")

    # 3. 查找底层 Model
    # 通常路径: executor -> driver_worker -> model_runner -> model
    try:
        if hasattr(executor, "driver_worker"):
            vllm_model = executor.driver_worker.model_runner.model
        elif hasattr(executor, "model_runner"):
            vllm_model = executor.model_runner.model
    except AttributeError:
        pass

    if vllm_model is None:
        raise RuntimeError("找到了 Executor 但无法定位 Model，请确保 enforce_eager=True")

    # 4. 执行更新 (In-place Copy)
    policy_params = dict(policy_model.named_parameters())
    
    with torch.no_grad():
        count = 0
        for name, vllm_param in vllm_model.named_parameters():
            if name in policy_params:
                # 确保数据在同一设备，执行原地拷贝
                vllm_param.copy_(policy_params[name])
                count += 1
            else:
                pass

def _vllm_worker_load_from_file(worker, file_path: str):
    import torch
    import os
    
    if not os.path.exists(file_path):
        return False

    if hasattr(worker, "model_runner"):
        model = worker.model_runner.model
        
        # 1. 从磁盘加载 (CPU)
        # map_location="cpu" 防止 worker 显存不足
        try:
            state_dict = torch.load(file_path, map_location="cpu")
        except Exception as e:
            print(f"[Worker] Load failed: {e}")
            return False
            
        # 2. 转换为 load_weights 需要的格式 [(name, tensor)]
        gpu_weights = []
        for name, tensor in state_dict.items():
            # 转换为 GPU Tensor
            gpu_weights.append((name, tensor.to("cuda")))
            
        # 3. 执行加载
        model.load_weights(gpu_weights)
        return True
        
    return False

def _vllm_worker_update_weights(worker, weights_dict):
    import torch
    
    if hasattr(worker, "model_runner"):
        model = worker.model_runner.model
        
        gpu_weights = []
        # 我们现在传的是字典 {name: list_of_floats}
        # 这样比 list of tuples 更不容易错位
        for name, data_list in weights_dict.items():
            # 安全检查
            if not isinstance(data_list, (list, tuple)):
                # 如果万一还是传错了，跳过以防崩坏
                continue
                
            # 核心：从纯数字列表重建 Tensor，并直接放到 GPU
            # dtype=model.dtype 确保精度一致 (如 bfloat16)
            # 如果不知道 dtype，可以先默认，或者不指定
            try:
                param_tensor = torch.tensor(data_list, device="cuda")
                # 如果维度对不上（比如 scalars），可能需要 reshape，但在 load_weights 里通常会自动处理
                gpu_weights.append((name, param_tensor))
            except Exception as e:
                # 打印错误但不崩溃
                print(f"Error reconstructing tensor for {name}: {e}")
                continue
        
        # 执行加载
        if gpu_weights:
            model.load_weights(gpu_weights)
        return True
    return False

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
    gpu_util = config["training"].get("gpu_memory_utilization", 0.4) # 默认改小
    
    llm = LLM(
        model=config["model"]["model_path"],
        dtype=config["model"]["dtype"],
        gpu_memory_utilization=gpu_util, 
        trust_remote_code=True,
        max_model_len=config["data"]["max_seq_length"], # 4096
        enforce_eager=True, # 显存优化技巧
        enable_prefix_caching=False,
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
    clip_range = config["training"]["clip_range"]
    
    start_step = config["model"]["start_step"]
    global_step = start_step if start_step is not None else 0
    pbar = tqdm(total=n_grpo_steps, desc="GRPO Steps")
    
    # 无限循环数据，直到达到 n_grpo_steps
    data_iter = iter(dataloader)
    best_reward = config["model"]["best_reward"]
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
        # sync_policy_to_vllm_inplace(policy, llm)
        # weights_to_send = {}
        # for name, param in policy.named_parameters():
        #     # 只发送需要梯度的参数（减少数据量，防止 RPC 超时）
        #     if param.requires_grad:
        #         # .cpu().tolist() 是最慢但最稳的方法，生成的全是 float
        #         weights_to_send[name] = param.data.cpu().tolist()
        sync_policy_to_vllm(policy, llm)

        # (可选) 打印确认
        # print(f"Synced weights via file: {temp_weight_path}")
        
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
            reward_fn=robust_reward_fn,
            rollout_responses=all_responses,
            repeated_ground_truths=all_ground_truths,
            group_size=config["training"]["group_size"],
            advantage_eps=config["training"]["advantage_eps"],
            normalize_by_std=True
        )
        # 转为 Tensor 并移到 GPU
        advantages = advantages.to(device).unsqueeze(1) # (B*G, 1)
        raw_rewards = raw_rewards.to(device).unsqueeze(1)

        format_scores = []
        answer_scores = []
        lengths = []
        
        for r, gt in zip(all_responses, all_ground_truths):
            # 重新调一次 reward_fn 仅仅为了记录 log (开销很小，字符串操作)
            # 或者修改 compute_group_norm 让它返回 dict list
            metrics = robust_reward_fn(r, gt)
            format_scores.append(metrics["format_reward"])
            answer_scores.append(metrics["answer_reward"])
            lengths.append(len(tokenizer.encode(r))) # 估算 Token 长度
        
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
        inference_batch_size = 4 
        old_log_probs_list = []
        
        policy.eval() # 切换到 Eval 模式更安全
        with torch.no_grad():
            for i in range(0, len(input_ids), inference_batch_size):
                batch_input_ids = input_ids[i : i + inference_batch_size]
                batch_labels = labels[i : i + inference_batch_size]
                
                log_probs_dict = get_response_log_probs(policy, batch_input_ids, batch_labels)
                old_log_probs_list.append(log_probs_dict["log_probs"].detach())
                
        old_log_probs = torch.cat(old_log_probs_list, dim=0)
        policy.train() # 恢复 Train 模式

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
        accum_stats = {"loss": 0.0, "clip_ratio": 0.0, "approx_kl": 0.0}
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
                
                loss, step_metrics = grpo_microbatch_train_step(
                    policy_log_probs=mb_policy_log_probs,
                    response_mask=mb_mask,
                    gradient_accumulation_steps=actual_accum_steps,
                    loss_type=config["training"]["loss_type"],
                    advantages=mb_adv,
                    old_log_probs=mb_old_lp,
                    cliprange=clip_range
                    # raw_rewards=mb_rewards
                )
                accum_stats["loss"] += loss.item() / actual_accum_steps
                accum_stats["clip_ratio"] += step_metrics["clip_ratio"].item() / actual_accum_steps
                accum_stats["approx_kl"] += step_metrics["approx_kl"].item() / actual_accum_steps
                
                step_loss += loss.item() / actual_accum_steps

            # End of Micro-batches -> Update
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config["training"]["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad()
        
        global_step += 1
        pbar.update(1)
        
        # --- Logging ---
        wandb.log({
            # 1. 核心表现
            "train/reward_mean": reward_meta["mean_reward"],
            "train/reward_std": raw_rewards.std().item(), # 组间方差
            "train/accuracy": np.mean(answer_scores),     # 真实准确率
            "train/format_rate": np.mean(format_scores),  # 格式正确率
            
            # 2. 训练动态
            "train/loss": accum_stats["loss"],
            "train/clip_fraction": accum_stats["clip_ratio"],
            "train/approx_kl": accum_stats["approx_kl"],
            "train/lr": optimizer.param_groups[0]['lr'],
            "train/grad_norm": grad_norm.item(), # 之前代码里算的
            
            # 3. 行为特征 (观察 R1 涌现的关键!)
            "train/completion_len_mean": np.mean(lengths),
            "train/completion_len_max": np.max(lengths),
            
            # Step
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
                reward_fn=robust_reward_fn,
                num_examples_to_log=config["evaluation"]["num_examples_to_log"],
                max_new_tokens=eval_max_tokens
            )
            
            # 合并日志
            eval_stats["train/global_step"] = global_step
            wandb.log(eval_stats, commit=False)
            
            policy.train() # 切回训练模式
            
        current_reward = reward_meta["mean_reward"]
        
        if current_reward > best_reward:
            best_reward = current_reward
            print(f"最佳reward: {global_step}时达到{best_reward:.4f}! 正在保存...")
            
            best_save_path = os.path.join(output_dir, "checkpoint-best")
            policy.save_pretrained(best_save_path)
            tokenizer.save_pretrained(best_save_path)
            
            # 在 WandB 里打个标记
            wandb.log({"train/best_reward": best_reward, "train/global_step": global_step}, commit=False)

        # B. 定期保存 (Periodic Save) - 防止意外中断
        # 建议在 config 里把 save_steps 设小一点，比如 50 或 100
        if global_step % config["training"]["save_steps"] == 0:
            print(f"定期保存中 {global_step}")
            
            # 保存路径带上步数
            step_save_path = os.path.join(output_dir, f"checkpoint-step-{global_step}")
            policy.save_pretrained(step_save_path)
            tokenizer.save_pretrained(step_save_path)


    print("训练完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo_config.yaml")
    args = parser.parse_args()
    train(args.config)