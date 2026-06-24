import argparse
import json
import os

from cs336_alignment.bootstrap_runtime import bootstrap_cuda_visible_devices

bootstrap_cuda_visible_devices(default_config_path="configs/train/sft_lora_config.yaml")

import torch
import wandb
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

from cs336_alignment.device_config import (
    apply_runtime_environment,
    build_model_load_kwargs,
    get_model_primary_device,
    resolve_torch_device,
)
from cs336_alignment.sft import get_response_log_probs, log_generations_transformer, sft_microbatch_train_step
from cs336_alignment.train_sft import SFTDataset, get_collate_fn
from cs336_alignment.utils import robust_reward_fn


def build_lora_model(config: dict, model_load_path: str, resume_from: str | None):
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:
        raise ImportError("LoRA 训练需要 `peft`。请先运行 `uv sync` 安装新增依赖。") from exc

    default_device = resolve_torch_device(config)
    model_load_kwargs = build_model_load_kwargs(config)
    model = AutoModelForCausalLM.from_pretrained(model_load_path, **model_load_kwargs)
    if "device_map" not in model_load_kwargs:
        model.to(default_device)

    if resume_from:
        print(f"从 LoRA adapter checkpoint 恢复: {resume_from}")
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            bias=config["lora"].get("bias", "none"),
            target_modules=config["lora"]["target_modules"],
        )
        model = get_peft_model(model, lora_cfg)

    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model.config.use_cache = False
    model.train()
    model.print_trainable_parameters()
    return model


def train(config_path: str, args):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    apply_runtime_environment(config)

    print(f"从{config_path}加载配置")
    print(json.dumps(config, indent=2))

    if args.wandb_id:
        print(f"Resuming WandB run: {args.wandb_id}")
        wandb.init(
            project=config["wandb"]["project"],
            id=args.wandb_id,
            resume="must",
            config=config,
        )
    else:
        wandb.init(
            project=config["wandb"]["project"],
            name=config["wandb"]["run_name"],
            config=config,
        )

    output_dir = config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    model_load_path = config["model"]["model_path"]

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_load_path,
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_lora_model(config, model_load_path, args.resume_from)
    device = get_model_primary_device(model)
    print(f"训练设备入口: {device}")

    prompt_path = config["data"]["prompt_path"]
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    max_samples = config["data"]["max_samples"]
    train_dataset = SFTDataset(config["data"]["train_path"], max_samples)
    max_len = config["data"].get("max_seq_length", 1024)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["micro_batch_size"],
        shuffle=True,
        collate_fn=get_collate_fn(tokenizer, max_len, prompt_template=prompt_template),
        drop_last=True,
    )

    valid_examples = []
    with open(config["data"]["valid_path"], "r", encoding="utf-8") as f:
        for line in f:
            valid_examples.append(json.loads(line))

    val_prompts = []
    for ex in valid_examples:
        val_prompts.append(prompt_template.replace("{question}", ex["problem"]))
    val_truths = [ex["solution"] for ex in valid_examples]

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=float(config["training"]["learning_rate"]))

    epochs = config["training"]["epochs"]
    grad_accum_steps = config["training"]["gradient_accumulation_steps"]
    clip_norm = config["training"]["max_grad_norm"]
    eval_every = config["evaluation"]["eval_every_steps"]

    total_micro_steps = 0
    accumulated_loss = 0.0
    steps_per_epoch = len(train_loader) // grad_accum_steps
    start_epoch = args.start_epoch
    global_step = start_epoch * steps_per_epoch
    total_global_steps = len(train_loader) * epochs // grad_accum_steps

    warmup_ratio = config["training"].get("warmup_ratio", 0)
    warmup_steps = int(total_global_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_global_steps,
        last_epoch=global_step - 1,
    )

    print("开始 LoRA SFT 训练")
    for epoch in range(start_epoch, epochs):
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            ncols=120,
            leave=True,
        )

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            log_probs_dict = get_response_log_probs(model, input_ids, attention_mask, labels)
            policy_log_probs = log_probs_dict["log_probs"]

            _, metrics = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=grad_accum_steps,
                normalize_constant=1,
            )

            accumulated_loss += metrics["loss"].item()
            total_micro_steps += 1

            if total_micro_steps % grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, clip_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                avg_loss = accumulated_loss / grad_accum_steps
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix(
                    {
                        "Step": f"{global_step}/{total_global_steps}",
                        "Loss": f"{avg_loss:.4f}",
                        "Norm": f"{grad_norm.item():.2f}",
                        "LR": f"{current_lr:.2e}",
                    }
                )

                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/grad_norm": grad_norm.item(),
                        "train/global_step": global_step,
                        "train/epoch": epoch + (progress_bar.n / len(train_loader)),
                        "train/lr": current_lr,
                    }
                )
                accumulated_loss = 0.0

                if global_step % eval_every == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    progress_bar.write(f"Step {global_step}: Running Evaluation...")
                    eval_stats = log_generations_transformer(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=val_prompts,
                        ground_truths=val_truths,
                        reward_fn=robust_reward_fn,
                        num_examples_to_log=config["evaluation"]["num_examples_to_log"],
                        max_new_tokens=config["evaluation"].get("max_new_tokens", 2048),
                    )
                    eval_stats["train/global_step"] = global_step
                    wandb.log(eval_stats)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    model.train()

        epoch_output_dir = os.path.join(output_dir, f"epoch{epoch}")
        print(f"正在保存 LoRA adapter 到 {epoch_output_dir}...")
        model.save_pretrained(epoch_output_dir)
        tokenizer.save_pretrained(epoch_output_dir)

    print("LoRA SFT 训练完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/sft_lora_config.yaml", help="Path to YAML config file")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a LoRA adapter checkpoint directory")
    parser.add_argument("--wandb_id", type=str, default=None, help="The wandb run id to resume")
    parser.add_argument("--start_epoch", type=int, default=0, help="The epoch number to start from")
    args = parser.parse_args()

    train(args.config, args)
