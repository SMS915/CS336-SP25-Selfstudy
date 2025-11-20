import argparse
import os
import torch
import json
import wandb
import yaml
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from torch.optim import AdamW

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft import get_response_log_probs, tokenize_prompt_and_output, sft_microbatch_train_step, log_generations

class SFTDataset(Dataset):
    def __init__(self, data_path, max_samples = None):
        self.data = []
        with open(data_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

        if max_samples is not None and max_samples > 0:
            self.data = self.data[:max_samples]
            print(f"截取 {len(self.data)} 条样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 返回原始字典: {"prompt": "...", "response": "..."}
        return self.data[idx]

def get_collate_fn(tokenizer, max_length = 1024, prompt_template = None):
    """
    闭包函数，为了把 tokenizer 传进去。
    DataLoader 会把 batch_data (list of dicts) 传给这个函数。
    """
    def collate_fn(batch_data):
        # 1. 解包数据
        prompts = []
        for item in batch_data:
            raw_prompt = item["prompt"]
            if prompt_template:
                # 替换 {question} 占位符
                p = prompt_template.replace("{question}", raw_prompt)
                prompts.append(p)
            else:
                prompts.append(raw_prompt)
        
        responses = [item["response"] for item in batch_data]
        # 2. 调用你写的核心 Tokenizer 函数
        # 这个函数已经处理了 Padding, Mask, Shift 等所有脏活累活
        tokenized_batch = tokenize_prompt_and_output(
            prompt_strs=prompts,
            output_strs=responses,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        return tokenized_batch
        
    return collate_fn

def train(config_path: str):
    # --- 加载配置 ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print(f"从{config_path}加载配置")
    print(json.dumps(config, indent=2))

    # --- WandB 初始化 ---
    wandb.init(
        project=config["wandb"]["project"],
        name=config["wandb"]["run_name"],
        config=config
    )

    # --- 路径与设备 ---
    output_dir = config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 加载模型与 Tokenizer ---
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["model_path"],
        torch_dtype=getattr(torch, config["model"]["dtype"]), # 动态获取 torch.bfloat16
        attn_implementation=config["model"]["attn_implementation"],
        device_map="auto"
    )
    model.gradient_checkpointing_enable()
    model.train()

    # --- 准备数据 ---
    prompt_path = config["data"]["prompt_path"]
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()
        f.close()
    max_samples = config["data"]["max_samples"]
    train_dataset = SFTDataset(config["data"]["train_path"], max_samples)
    max_len = config["data"].get("max_seq_length", 1024)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["micro_batch_size"],
        shuffle=True,
        collate_fn=get_collate_fn(tokenizer, max_len, prompt_template = prompt_template),
        drop_last=True
    )
    valid_examples = []
    with open(config["data"]["valid_path"], "r") as f:
        for line in f:
            valid_examples.append(json.loads(line))
    # 加载验证集用于 Log Generation
    val_prompts = []
    for ex in valid_examples:
        # 这里假设你在函数开头已经加载了 prompt_template 字符串
        # 如果 prompt_template 是 None，请确保先读取它
        formatted_prompt = prompt_template.replace("{question}", ex["problem"])
        val_prompts.append(formatted_prompt)
        
    val_truths = [ex["solution"] for ex in valid_examples]

    # --- 优化器 ---
    optimizer = AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))

    # --- 训练循环变量 ---
    epochs = config["training"]["epochs"]
    grad_accum_steps = config["training"]["gradient_accumulation_steps"]
    clip_norm = config["training"]["max_grad_norm"]
    eval_every = config["evaluation"]["eval_every_steps"]
    
    global_step = 0
    total_micro_steps = 0
    
    print("开始训练")
    
    for epoch in range(epochs):
        # 使用 tqdm 包装 loader 显示进度
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # 1. 搬运数据
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)

            # 2. 获取 Log Probs
            # 注意：return_token_entropy=False 节省计算，SFT loss 不用它
            log_probs_dict = get_response_log_probs(model, input_ids, labels)
            policy_log_probs = log_probs_dict["log_probs"]
            
            # 3. 计算 Loss 并 Backward
            # sft_microbatch_train_step 内部会处理 loss / accum_steps
            # 并执行 .backward()
            loss, metrics = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=grad_accum_steps,
                normalize_constant=1
            )
            
            total_micro_steps += 1
            
            # 更新进度条上的 Loss 显示
            progress_bar.set_postfix(loss=metrics["loss"].item())

            # 4. 梯度累积更新
            if total_micro_steps % grad_accum_steps == 0:
                # 裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                
                # 更新
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # 记录日志 (记录真实的 batch loss)
                wandb.log({
                    "train/loss": metrics["loss"].item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/global_step": global_step,
                    "train/epoch": epoch + (total_micro_steps / len(train_loader))
                })

                # 5. 评估 (抽查生成)
                if global_step % eval_every == 0:
                    # 临时释放显存压力（如果有必要的话，但在 4090 上 pytorch 原生生成应该还好）
                    # torch.cuda.empty_cache()
                    
                    eval_stats = log_generations(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=val_prompts,
                        ground_truths=val_truths,
                        reward_fn=r1_zero_reward_fn,
                        num_examples_to_log=config["evaluation"]["num_examples_to_log"]
                    )
                    
                    # 合并日志
                    eval_stats["train/global_step"] = global_step
                    wandb.log(eval_stats)
                    
                    model.train() # 切回训练模式

    # --- 保存最终模型 ---
    print(f"正在保存模型到{output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("训练完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    
    train(args.config)