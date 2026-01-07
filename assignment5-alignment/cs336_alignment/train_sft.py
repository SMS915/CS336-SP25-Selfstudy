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
from transformers.optimization import get_cosine_schedule_with_warmup 

from cs336_alignment.sft import get_response_log_probs, sft_microbatch_train_step, log_generations
from cs336_alignment.utils import tokenize_prompt_and_output, robust_reward_fn

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

        tokenized_batch = tokenize_prompt_and_output(
            prompt_strs=prompts,
            output_strs=responses,
            tokenizer=tokenizer,
            max_length=max_length,
            sft_train=True
        )
        # if not hasattr(collate_fn, "has_printed"):
        #     input_ids = tokenized_batch["input_ids"][0] # 取第一个样本
        #     labels = tokenized_batch["labels"][0]
            
        #     print("\n" + "!"*30 + " CRITICAL DEBUG " + "!"*30)
            
        #     # 1. 检查 Special Token ID 是否存在于 Input
        #     ans_end_id = tokenizer.convert_tokens_to_ids("</answer>")
        #     print(f"Target Special ID: {ans_end_id}")
            
        #     if ans_end_id in input_ids:
        #         print(f"✅ Input_ids contains {ans_end_id}")
        #     else:
        #         print(f"❌ Input_ids DOES NOT contain {ans_end_id} !!!")
        #         print("Top 10 tokens:", input_ids[:10])
        #         print("Last 10 tokens:", input_ids[-10:])
            
        #     # 2. 检查 Label 是否 Mask 了它
        #     # 找到 </answer> 在 input_ids 的位置
        #     try:
        #         # 找最后一次出现的位置
        #         loc = (input_ids == ans_end_id).nonzero(as_tuple=True)[0][-1].item()
        #         # label 是 shift 过的，所以对应位置的 label 应该就是 ans_end_id
        #         # 注意：labels[i] 对应 input_ids[i+1] (因为 shift)
        #         # 你的 tokenize 函数返回的 labels 已经是 shift 过的了
                
        #         # 检查 labels 对应位置是否是 -100
        #         # 你的代码: labels = batch_input_ids[:, 1:]
        #         # 你的代码: input_ids = batch_input_ids[:, :-1]
        #         # 这意味着 labels[i] 实际上是 input_ids[i] 的下一个词
                
        #         print(f"Label at pos {loc} (should be EOS or next): {labels[loc]}")
        #         print(f"Label at pos {loc-1} (should be </answer>): {labels[loc-1]}")
                
        #     except IndexError:
        #         pass
            
        #     collate_fn.has_printed = True
        #     print("!"*80 + "\n")
        #     # 如果发现没有 ID，直接报错停止，别练了
        #     if ans_end_id not in input_ids:
        #         raise RuntimeError("FATAL: Tokenizer failed to encode </answer> as a single ID!")
        
        return tokenized_batch
        
    return collate_fn

def train(config_path: str, args):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print(f"从{config_path}加载配置")
    print(json.dumps(config, indent=2))

    # WandB 初始化
    if args.wandb_id:
        print(f"Resuming WandB run: {args.wandb_id}")
        wandb.init(
            project=config["wandb"]["project"],
            id=args.wandb_id,   # 指定 ID
            resume="must",      # 强制续训，如果ID不存在会报错
            config=config
        )
    else:
        wandb.init(
            project=config["wandb"]["project"],
            name=config["wandb"]["run_name"],
            config=config
        )

    # 路径与设备
    output_dir = config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_load_path = args.resume_from if args.resume_from else config["model"]["model_path"]

    # 加载模型与 Tokenizer
    special_thinking_tag = {'additional_special_tokens': ['<think>', '</think>', '<answer>', '</answer>']}

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # num_add_tokens = tokenizer.add_special_tokens(special_thinking_tag)
    # print(f"添加了{num_add_tokens}个特殊token")
        
    model = AutoModelForCausalLM.from_pretrained(
        model_load_path,
        torch_dtype=getattr(torch, config["model"]["dtype"]), # 动态获取 torch.bfloat16
        attn_implementation=config["model"]["attn_implementation"],
        device_map="auto"
    )

    # if num_add_tokens != 0:
    #     model.resize_token_embeddings(len(tokenizer))
    #     with torch.no_grad():
    #         input_embed = model.get_input_embeddings().weight
    #         if not model.config.tie_word_embeddings:
    #             output_embed = model.get_output_embeddings().weight
    #         token_map = {'<think>': 'think', '</think>': '<|endoftext|>', '<answer>': 'answer', '</answer>': '<|endoftext|>'}
    #         for special_token, reference in token_map.items():
    #             special_id = tokenizer.convert_tokens_to_ids(special_token)
    #             ref_id = tokenizer.convert_tokens_to_ids(reference)
    #             input_embed[special_id] = input_embed[ref_id].clone() if reference != '<|endoftext|>' else input_embed[ref_id] + torch.randn_like(input_embed[ref_id]) * 0.01
    #             print(f"已用{reference}的语义初始化{special_token}")
    #             if not model.config.tie_word_embeddings:
    #                 output_embed[special_id] = output_embed[ref_id].clone() if reference != '<|endoftext|>' else output_embed[ref_id] + torch.randn_like(output_embed[ref_id]) * 0.01
    #     added_output_dir = os.path.join(output_dir, 'added_special_token')
    #     model.save_pretrained(added_output_dir)
    #     tokenizer.save_pretrained(added_output_dir)


    # if num_add_tokens != 0:
    #     model.resize_token_embeddings(len(tokenizer))
    #     with torch.no_grad():
    #         tied = model.config.tie_word_embeddings
    #         input_embed = model.get_input_embeddings()
    #         if not tied:
    #             output_embed = model.get_output_embeddings()
    #         new_token_embeddings = input_embed.weight.data[-num_add_tokens:]
    #         print(f"随机初始化后的新token embedding (前5个值): \n{new_token_embeddings[0, :5]}")
    #         mean_input_embed = input_embed.weight.data[:-num_add_tokens].mean(dim=0)
    #         if not tied:
    #             mean_output_embed = model.get_output_embeddings().weight.data[:-num_add_tokens].mean(dim=0)
    #         print(f"计算出的均值embedding (前5个值): \n{mean_input_embed[:5]}")
    #         for i in range(num_add_tokens):
    #             input_embed.weight.data[-num_add_tokens + i] = mean_input_embed
    #             if not tied:
    #                 output_embed.weight.data[-num_add_tokens + i] = mean_output_embed
    #     added_output_dir = os.path.join(output_dir, 'added_special_token')
    #     model.save_pretrained(added_output_dir)
    #     tokenizer.save_pretrained(added_output_dir)

    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    tokenizer_vocab_size = len(tokenizer)
    # assert model_vocab_size == tokenizer_vocab_size, "tokenizer与model的词表大小不匹配"

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()

    # 准备数据
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
        formatted_prompt = prompt_template.replace("{question}", ex["problem"])
        val_prompts.append(formatted_prompt)
        
    val_truths = [ex["solution"] for ex in valid_examples]

    # 优化器
    optimizer = AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))

    # --- 训练循环变量 ---
    epochs = config["training"]["epochs"]
    grad_accum_steps = config["training"]["gradient_accumulation_steps"]
    clip_norm = config["training"]["max_grad_norm"]
    eval_every = config["evaluation"]["eval_every_steps"]
    save_interval = config["training"]["save_steps"]
    
    global_step = 0
    total_micro_steps = 0
    accumulated_loss = 0.0
    
    total_global_steps = len(train_loader) * epochs // grad_accum_steps
    print("开始训练")
    start_epoch = args.start_epoch
    steps_per_epoch = len(train_loader) // grad_accum_steps
    global_step = start_epoch * steps_per_epoch 

    warmup_ratio = config["training"].get("warmup_ratio", 0)
    warmup_steps = int(total_global_steps * warmup_ratio)
    print(f"Total Global Steps: {total_global_steps}, Warmup Steps: {warmup_steps}")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_global_steps,
        last_epoch=global_step - 1 
    )


    for epoch in range(start_epoch, epochs):
        # 使用 tqdm 包装 loader 显示进度
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{epochs}",
            ncols=120, # 限制宽度，防止在某些终端换行
            leave=True 
        )

        epoch_loss_tracker = 0.0
        
        for batch in progress_bar:
            # 1. 搬运数据
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # 2. 获取 Log Probs
            # 注意：return_token_entropy=False 节省计算，SFT loss 不用它
            log_probs_dict = get_response_log_probs(model, input_ids, attention_mask, labels)
            policy_log_probs = log_probs_dict["log_probs"]
            
            # 3. 计算 Loss 并 Backward
            _, metrics = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=grad_accum_steps,
                normalize_constant=1
            )
            
            accumulated_loss += metrics["loss"].item()
            total_micro_steps += 1

            # 4. 梯度累积更新
            if total_micro_steps % grad_accum_steps == 0:
                # 裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                
                scheduler.step()

                # 更新
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1

                avg_loss = accumulated_loss / grad_accum_steps
                epoch_loss_tracker += avg_loss # 记录一下
                current_lr = scheduler.get_last_lr()[0]
                # 记录日志 (记录真实的 batch loss)
                progress_bar.set_postfix({
                                    "Step": f"{global_step}/{total_global_steps}", # 显示全局进度
                                    "Loss": f"{avg_loss:.4f}",    # 显示真实的 step loss
                                    "Norm": f"{grad_norm.item():.2f}", # 监控梯度爆炸
                                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}" # 监控 LR 变化
                                })
                
                wandb.log({
                    "train/loss": avg_loss,
                    "train/grad_norm": grad_norm.item(),
                    "train/global_step": global_step,
                    "train/epoch":  epoch + (progress_bar.n / len(train_loader)),
                    "train/lr": current_lr
                })

                accumulated_loss = 0.0

                # 5. 评估 (抽查生成)
                if global_step % eval_every == 0:
                    torch.cuda.empty_cache()
                    progress_bar.write(f"Step {global_step}: Running Evaluation...")
                    eval_max_tokens = config["evaluation"].get("max_new_tokens", 2048)
                    eval_stats = log_generations(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=val_prompts,
                        ground_truths=val_truths,
                        reward_fn=robust_reward_fn,
                        num_examples_to_log=config["evaluation"]["num_examples_to_log"],
                        max_new_tokens=eval_max_tokens
                    )
                    
                    # 合并日志
                    eval_stats["train/global_step"] = global_step
                    wandb.log(eval_stats)
                    del eval_stats
                    torch.cuda.empty_cache() 
                    model.train() # 切回训练模式

        # 保存该epoch模型
        epoch_output_dir = os.path.join(output_dir, f'epoch{epoch}')
        print(f"正在保存模型到{epoch_output_dir}...")
        model.save_pretrained(epoch_output_dir)
        tokenizer.save_pretrained(epoch_output_dir)
        
    print("训练完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml", help="Path to YAML config file")

    parser.add_argument("--resume_from", type=str, default=None, help="Path to the checkpoint directory (e.g., checkpoints/sft_v1/epoch0)")
    parser.add_argument("--wandb_id", type=str, default=None, help="The ID of the wandb run to resume (e.g., a1b2c3d4)")
    parser.add_argument("--start_epoch", type=int, default=0, help="The epoch number to start form (e.g., 1)")
    args = parser.parse_args()
    
    train(args.config, args)