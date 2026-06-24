import argparse
import json
import math
import os
from typing import Dict, List

from cs336_alignment.bootstrap_runtime import bootstrap_cuda_visible_devices

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

bootstrap_cuda_visible_devices(default_config_path="configs/train/sft_fsdp_qwen3_config.yaml")

import torch
import wandb
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

from cs336_alignment.device_config import apply_runtime_environment, build_model_load_kwargs
from cs336_alignment.fsdp_utils import (
    barrier,
    destroy_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    maybe_no_sync,
    rank0_print,
    reduce_metrics,
    save_fsdp_model,
    wrap_model_with_fsdp,
)
from cs336_alignment.process_title import set_python_process_title
from cs336_alignment.sft import get_response_log_probs, sft_microbatch_train_step
from cs336_alignment.utils import tokenize_prompt_and_output


class SFTDataset(Dataset):
    def __init__(self, data_path: str, max_samples: int | None = None):
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        if max_samples is not None and max_samples > 0:
            self.data = self.data[:max_samples]
            rank0_print(f"截取 {len(self.data)} 条样本")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.data[idx]


def _extract_prompt_text(item: Dict[str, str]) -> str | None:
    return item.get("prompt") or item.get("question") or item.get("problem")


def _extract_response_text(item: Dict[str, str]) -> str | None:
    return item.get("response") or item.get("solution") or item.get("answer")


def get_collate_fn(tokenizer, max_length: int = 1024, prompt_template: str | None = None):
    def collate_fn(batch_data: List[Dict[str, str]]):
        prompts = []
        responses = []
        for item in batch_data:
            raw_prompt = _extract_prompt_text(item)
            raw_response = _extract_response_text(item)
            if raw_prompt is None or raw_response is None:
                raise KeyError("SFT batch item must contain prompt/question/problem and response/solution/answer")
            prompt = prompt_template.replace("{question}", raw_prompt) if prompt_template else raw_prompt
            prompts.append(prompt)
            responses.append(raw_response)

        return tokenize_prompt_and_output(
            prompt_strs=prompts,
            output_strs=responses,
            tokenizer=tokenizer,
            max_length=max_length,
            sft_train=True,
        )

    return collate_fn


def build_validation_loader(config: Dict, tokenizer, prompt_template: str | None):
    valid_path = config["data"].get("valid_path")
    if not valid_path:
        return None

    valid_dataset = SFTDataset(valid_path, config["data"].get("valid_max_samples"))
    val_sampler = DistributedSampler(valid_dataset, shuffle=False, drop_last=False)
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config["evaluation"].get("micro_batch_size", 1),
        sampler=val_sampler,
        collate_fn=get_collate_fn(
            tokenizer,
            config["data"].get("max_seq_length", 1024),
            prompt_template=prompt_template,
        ),
        drop_last=False,
    )
    return val_loader


def evaluate_validation_loss(model, val_loader, device: torch.device, dtype: torch.dtype, max_batches: int) -> float:
    if val_loader is None:
        return float("nan")

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                log_probs_dict = get_response_log_probs(model, input_ids, attention_mask, labels)
                per_token_loss = -log_probs_dict["log_probs"]
                masked_loss = per_token_loss * response_mask
                valid_tokens = response_mask.sum().item()
                batch_loss = masked_loss.sum().item() / max(valid_tokens, 1)

            total_loss += batch_loss
            total_batches += 1

    if was_training:
        model.train()

    if total_batches == 0:
        return float("nan")
    return total_loss / total_batches


def train(config_path: str, args):
    set_python_process_title()
    local_rank, _, world_size = init_distributed()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    apply_runtime_environment(config)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    dtype = getattr(torch, config["model"].get("dtype", "bfloat16"))

    rank0_print(f"从 {config_path} 加载配置")
    if is_main_process():
        print(json.dumps(config, indent=2, ensure_ascii=False))

    run = None
    if is_main_process():
        if args.wandb_id:
            run = wandb.init(
                project=config["wandb"]["project"],
                id=args.wandb_id,
                resume="must",
                config=config,
            )
        else:
            run = wandb.init(
                project=config["wandb"]["project"],
                name=config["wandb"]["run_name"],
                config=config,
            )

    model_load_path = args.resume_from if args.resume_from else config["model"]["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_load_path,
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_template = None
    prompt_path = config["data"].get("prompt_path")
    if prompt_path:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

    model_load_kwargs = build_model_load_kwargs(config)
    model_load_kwargs.pop("device_map", None)
    model_load_kwargs.pop("max_memory", None)
    model = AutoModelForCausalLM.from_pretrained(model_load_path, **model_load_kwargs)
    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()

    model = wrap_model_with_fsdp(model, config, local_rank)

    train_dataset = SFTDataset(config["data"]["train_path"], config["data"].get("max_samples"))
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["micro_batch_size"],
        sampler=train_sampler,
        collate_fn=get_collate_fn(
            tokenizer,
            config["data"].get("max_seq_length", 1024),
            prompt_template=prompt_template,
        ),
        drop_last=True,
    )
    val_loader = build_validation_loader(config, tokenizer, prompt_template)

    optimizer = AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))
    grad_accum_steps = int(config["training"]["gradient_accumulation_steps"])
    epochs = int(config["training"]["epochs"])
    save_steps = int(config["training"]["save_steps"])
    eval_every = int(config["evaluation"].get("eval_every_steps", 0))
    val_max_batches = int(config["evaluation"].get("max_eval_batches", 8))
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))

    steps_per_epoch = max(math.ceil(len(train_loader) / grad_accum_steps), 1)
    total_global_steps = max(steps_per_epoch * epochs, 1)
    warmup_ratio = float(config["training"].get("warmup_ratio", 0.0))
    warmup_steps = int(total_global_steps * warmup_ratio)
    global_step = args.start_epoch * steps_per_epoch

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_global_steps,
        last_epoch=global_step - 1,
    )

    output_dir = config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    rank0_print(
        f"开始 FSDP SFT 训练: world_size={world_size}, "
        f"local_micro_batch={config['training']['micro_batch_size']}, grad_accum={grad_accum_steps}"
    )

    total_micro_steps = 0
    accumulation_micro_steps = 0
    accumulated_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [rank {get_rank()}]",
            ncols=120,
            disable=not is_main_process(),
        )

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            total_micro_steps += 1
            accumulation_micro_steps += 1
            is_last_batch = batch_idx == len(train_loader) - 1
            should_sync = accumulation_micro_steps >= grad_accum_steps or is_last_batch
            with maybe_no_sync(model, not should_sync):
                with torch.amp.autocast(device_type=device.type, dtype=dtype):
                    log_probs_dict = get_response_log_probs(model, input_ids, attention_mask, labels)
                    policy_log_probs = log_probs_dict["log_probs"]
                    _, metrics = sft_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=response_mask,
                        gradient_accumulation_steps=grad_accum_steps,
                        normalize_constant=1,
                    )

            accumulated_loss += metrics["loss"].item()

            if should_sync:
                grad_norm = model.clip_grad_norm_(max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                reduced = reduce_metrics(
                    {
                        "train/loss": accumulated_loss / accumulation_micro_steps,
                        "train/grad_norm": float(grad_norm.item()),
                        "train/lr": float(scheduler.get_last_lr()[0]),
                    },
                    device,
                )
                accumulated_loss = 0.0
                accumulation_micro_steps = 0

                if is_main_process():
                    progress_bar.set_postfix(
                        {
                            "Step": f"{global_step}/{total_global_steps}",
                            "Loss": f"{reduced['train/loss']:.4f}",
                            "Norm": f"{reduced['train/grad_norm']:.2f}",
                            "LR": f"{reduced['train/lr']:.2e}",
                        }
                    )
                    if run is not None:
                        wandb.log(
                            {
                                **reduced,
                                "train/global_step": global_step,
                                "train/epoch": epoch + (progress_bar.n / max(len(train_loader), 1)),
                            }
                        )

                if eval_every > 0 and global_step % eval_every == 0:
                    val_loss = evaluate_validation_loss(model, val_loader, device, dtype, val_max_batches)
                    reduced_val_loss = reduce_metrics({"eval/val_loss": val_loss}, device)
                    if is_main_process():
                        rank0_print(f"Step {global_step}: val_loss={reduced_val_loss['eval/val_loss']:.4f}")
                        if run is not None:
                            wandb.log(
                                {
                                    **reduced_val_loss,
                                    "train/global_step": global_step,
                                }
                            )

                if save_steps > 0 and global_step % save_steps == 0:
                    step_output_dir = os.path.join(output_dir, f"checkpoint-step-{global_step}")
                    rank0_print(f"保存 FSDP SFT checkpoint: {step_output_dir}")
                    save_fsdp_model(model, tokenizer, step_output_dir)

        epoch_output_dir = os.path.join(output_dir, f"epoch{epoch + 1}")
        rank0_print(f"保存 epoch checkpoint: {epoch_output_dir}")
        save_fsdp_model(model, tokenizer, epoch_output_dir)

    if run is not None:
        run.finish()
    destroy_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/sft_fsdp_qwen3_config.yaml")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    args = parser.parse_args()

    train(args.config, args)
