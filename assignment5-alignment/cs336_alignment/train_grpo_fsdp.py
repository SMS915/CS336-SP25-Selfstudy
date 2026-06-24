import argparse
import json
import math
import os
import queue
import random
import re
from typing import Any, Callable, Dict, List

from cs336_alignment.bootstrap_runtime import bootstrap_cuda_visible_devices

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

bootstrap_cuda_visible_devices(default_config_path="configs/train/grpo_fsdp_qwen3_config.yaml")

import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
import yaml
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

from cs336_alignment.device_config import apply_runtime_environment, build_model_load_kwargs
from cs336_alignment.drgrpo_grader import extract_answer, qwen_instruct_reward_fn, question_only_reward_fn
from cs336_alignment.fsdp_utils import (
    destroy_distributed,
    gather_full_state_dict,
    get_rank,
    get_wrapped_model,
    get_world_size,
    init_distributed,
    is_main_process,
    maybe_no_sync,
    rank0_print,
    reduce_metrics,
    save_fsdp_model,
    wrap_model_with_fsdp,
)
from cs336_alignment.grpo import compute_group_normalized_rewards, grpo_microbatch_train_step
from cs336_alignment.process_title import set_python_process_title
from cs336_alignment.sft import get_response_log_probs
from cs336_alignment.utils import robust_reward_fn, tokenize_prompt_and_output


class PromptAnswerDataset(Dataset):
    def __init__(self, data_path: str, prompt_template: str | None = None, max_samples: int | None = None):
        self.prompts: List[str] = []
        self.ground_truths: List[str] = []

        with open(data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_samples is not None and max_samples > 0 and len(self.prompts) >= max_samples:
                    break

                item = json.loads(line)
                raw_prompt = item.get("problem") or item.get("question") or item.get("prompt")
                raw_truth = item.get("answer") or item.get("solution")

                if raw_truth is None and item.get("response"):
                    raw_truth = self._extract_answer_from_response(str(item["response"]))

                if not raw_prompt or not raw_truth:
                    continue

                prompt = prompt_template.replace("{question}", str(raw_prompt).strip()) if prompt_template else str(raw_prompt).strip()
                self.prompts.append(prompt)
                self.ground_truths.append(str(raw_truth).strip())

        rank0_print(f"加载完成: {len(self.prompts)} 条样本用于 FSDP RL.")

    def _extract_answer_from_response(self, text: str) -> str | None:
        if "<answer>" in text and "</answer>" in text:
            match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
            if match:
                return match.group(1).strip()

        extracted = extract_answer(text)
        if extracted is None:
            return None
        return str(extracted).strip()

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {
            "prompt": self.prompts[idx],
            "ground_truth": self.ground_truths[idx],
        }


def collate_prompt_batch(batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
    return {
        "prompt": [item["prompt"] for item in batch],
        "ground_truth": [item["ground_truth"] for item in batch],
    }


def build_reward_wrapper(reward_style: str) -> Callable[[str, str, bool], Dict[str, float]]:
    normalized = reward_style.lower()
    if normalized == "deepseek_r1":
        return lambda response, truth, length_panelty=False: robust_reward_fn(
            response,
            truth,
            length_panelty=length_panelty,
        )
    if normalized == "boxed_only":
        return lambda response, truth, length_panelty=False: question_only_reward_fn(response, truth)
    if normalized == "qwen_boxed":
        return lambda response, truth, length_panelty=False: qwen_instruct_reward_fn(response, truth)
    raise ValueError(f"Unsupported reward_style: {reward_style}")


def postprocess_generation(text: str, stop_strings: List[str]) -> str:
    processed = text
    for stop_str in stop_strings:
        idx = processed.find(stop_str)
        if idx != -1:
            processed = processed[: idx + len(stop_str)]
            break
    return processed.strip()


def set_generation_mode(model: torch.nn.Module, enabled: bool) -> None:
    base_model = get_wrapped_model(model)
    if enabled:
        base_model.config.use_cache = True
        if hasattr(base_model, "gradient_checkpointing_disable"):
            base_model.gradient_checkpointing_disable()
    else:
        base_model.config.use_cache = False
        if hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable()


def generate_with_fsdp_full_params(
    model: torch.nn.Module,
    **generate_kwargs,
):
    base_model = get_wrapped_model(model)
    if isinstance(model, FSDP):
        # HF generate is proxied to the wrapped model, so we need to materialize
        # full params first; otherwise modules such as embeddings still see the
        # flattened local shard and crash with "'weight' must be 2-D".
        with FSDP.summon_full_params(model, recurse=True, writeback=False):
            return base_model.generate(**generate_kwargs)
    return base_model.generate(**generate_kwargs)


def build_eval_examples(
    config: Dict[str, Any],
    prompt_template: str | None,
) -> tuple[List[str], List[str]]:
    evaluation_cfg = config.get("evaluation", {})
    eval_path = evaluation_cfg.get("valid_path") or config["data"].get("valid_path")
    if not eval_path:
        raise ValueError("启用 evaluation 时必须提供 evaluation.valid_path 或 data.valid_path")

    eval_dataset = PromptAnswerDataset(
        eval_path,
        prompt_template=prompt_template,
        max_samples=evaluation_cfg.get("num_eval_examples"),
    )
    if len(eval_dataset) == 0:
        raise ValueError(f"评估数据为空: {eval_path}")
    return eval_dataset.prompts, eval_dataset.ground_truths


def evaluate_model_samples(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str, bool], Dict[str, float]],
    stop_strings: List[str],
    eval_cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    set_generation_mode(model, enabled=True)
    model.eval()

    batch_size = int(eval_cfg.get("eval_batch_size", 1))
    num_rollouts = int(eval_cfg.get("num_rollouts_per_example", 1))
    top_p = float(eval_cfg.get("top_p", 0.95))
    total_rewards: List[float] = []
    format_rewards: List[float] = []
    answer_rewards: List[float] = []
    completion_lengths: List[int] = []

    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            batch_truths = ground_truths[start : start + batch_size]

            encoded = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(device)
            prompt_lengths = encoded.attention_mask.sum(dim=-1).tolist()
            generation_output = model.generate(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                do_sample=True,
                temperature=float(eval_cfg.get("temperature", 0.6)),
                top_p=top_p,
                min_new_tokens=int(eval_cfg.get("sampling_min_tokens", 4)),
                max_new_tokens=int(eval_cfg.get("max_new_tokens", 1024)),
                num_return_sequences=num_rollouts,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

            sequences = generation_output.sequences
            for prompt_idx, truth in enumerate(batch_truths):
                for rollout_idx in range(num_rollouts):
                    seq_idx = prompt_idx * num_rollouts + rollout_idx
                    generated_ids = sequences[seq_idx][prompt_lengths[prompt_idx]:]
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    processed_text = postprocess_generation(generated_text, stop_strings)
                    metrics = reward_fn(processed_text, truth, False)

                    total_rewards.append(float(metrics.get("reward", 0.0)))
                    format_rewards.append(float(metrics.get("format_reward", 0.0)))
                    answer_rewards.append(float(metrics.get("answer_reward", 0.0)))
                    if tokenizer.pad_token_id is not None:
                        completion_lengths.append(int((generated_ids != tokenizer.pad_token_id).sum().item()))
                    else:
                        completion_lengths.append(len(generated_ids))

    return {
        "eval/reward_mean": float(np.mean(total_rewards)),
        "eval/reward_std": float(np.std(total_rewards)),
        "eval/format_rate": float(np.mean(format_rewards)),
        "eval/answer_rate": float(np.mean(answer_rewards)),
        "eval/completion_len_mean": float(np.mean(completion_lengths)),
        "eval/completion_len_max": float(np.max(completion_lengths)),
        "eval/num_examples": float(len(prompts)),
        "eval/num_rollouts_per_example": float(num_rollouts),
        "eval/num_samples": float(len(total_rewards)),
    }


def evaluation_worker_main(
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    worker_config: Dict[str, Any],
) -> None:
    set_python_process_title()
    torch.set_grad_enabled(False)

    seed = int(worker_config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(worker_config["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)

    load_config = {
        "model": {
            "dtype": worker_config["dtype"],
            "trust_remote_code": worker_config.get("trust_remote_code", True),
            "attn_implementation": worker_config.get("attn_implementation"),
        },
        "runtime": {
            "device": str(device),
        },
    }
    model_load_kwargs = build_model_load_kwargs(load_config)
    model_load_kwargs.pop("device_map", None)
    model_load_kwargs.pop("max_memory", None)

    tokenizer = AutoTokenizer.from_pretrained(
        worker_config["model_path"],
        trust_remote_code=worker_config.get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(worker_config["model_path"], **model_load_kwargs)
    model.to(device)
    model.eval()

    reward_fn = build_reward_wrapper(worker_config["reward_style"])
    stop_strings = worker_config.get("stop_strings", [])

    while True:
        task = request_queue.get()
        if task is None:
            break

        step = int(task["step"])
        try:
            step_seed = seed + step
            random.seed(step_seed)
            np.random.seed(step_seed)
            torch.manual_seed(step_seed)

            model.load_state_dict(task["state_dict"], strict=True)
            metrics = evaluate_model_samples(
                model=model,
                tokenizer=tokenizer,
                prompts=worker_config["prompts"],
                ground_truths=worker_config["ground_truths"],
                reward_fn=reward_fn,
                stop_strings=stop_strings,
                eval_cfg=worker_config["evaluation"],
                device=device,
            )
            response_queue.put({"step": step, "metrics": metrics, "error": None})
        except Exception as exc:  # pragma: no cover - worker failures are surfaced to the trainer
            response_queue.put({"step": step, "metrics": None, "error": repr(exc)})


def start_evaluation_worker(
    config: Dict[str, Any],
    eval_prompts: List[str],
    eval_truths: List[str],
    stop_strings: List[str],
) -> tuple[mp.Process, mp.Queue, mp.Queue]:
    evaluation_cfg = config["evaluation"]
    eval_gpu_index = int(evaluation_cfg.get("eval_gpu_index", 0))
    visible_devices = str(config.get("runtime", {}).get("cuda_visible_devices") or "").strip()
    eval_cuda_visible_devices: str | None = None
    if visible_devices:
        visible_device_list = [item.strip() for item in visible_devices.split(",") if item.strip()]
        visible_device_count = len(visible_device_list)
        if eval_gpu_index >= visible_device_count:
            raise ValueError(
                f"evaluation.eval_gpu_index={eval_gpu_index} 超出 CUDA_VISIBLE_DEVICES={visible_devices} 的可见范围"
            )
        eval_cuda_visible_devices = visible_device_list[eval_gpu_index]
    worker_config = {
        "model_path": config["model"]["model_path"],
        "dtype": config["model"].get("dtype", "bfloat16"),
        "trust_remote_code": config["model"].get("trust_remote_code", True),
        "attn_implementation": config["model"].get("attn_implementation"),
        "cuda_visible_devices": eval_cuda_visible_devices if eval_cuda_visible_devices is not None else str(eval_gpu_index),
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "reward_style": config["training"].get("reward_style", "boxed_only"),
        "stop_strings": stop_strings,
        "prompts": eval_prompts,
        "ground_truths": eval_truths,
        "seed": int(config.get("runtime", {}).get("seed", 0)),
        "evaluation": {
            "eval_batch_size": int(evaluation_cfg.get("eval_batch_size", 1)),
            "num_rollouts_per_example": int(evaluation_cfg.get("num_rollouts_per_example", 1)),
            "temperature": float(evaluation_cfg.get("temperature", config["training"].get("sampling_temperature", 0.6))),
            "top_p": float(evaluation_cfg.get("top_p", config["training"].get("top_p", 0.95))),
            "sampling_min_tokens": int(
                evaluation_cfg.get("sampling_min_tokens", config["training"].get("sampling_min_tokens", 4))
            ),
            "max_new_tokens": int(
                evaluation_cfg.get("max_new_tokens", config["training"].get("sampling_max_tokens", 1024))
            ),
        },
    }

    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue(maxsize=1)
    response_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=evaluation_worker_main,
        args=(request_queue, response_queue, worker_config),
        daemon=True,
    )
    original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        if worker_config.get("cuda_visible_devices") is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_config["cuda_visible_devices"])
        process.start()
    finally:
        if original_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
    return process, request_queue, response_queue


def maybe_broadcast_bool(value: bool, device: torch.device) -> bool:
    tensor = torch.tensor(1 if value else 0, device=device)
    if get_world_size() > 1:
        torch.distributed.broadcast(tensor, src=0)
    return bool(tensor.item())


def train(config_path: str):
    set_python_process_title()
    local_rank, _, world_size = init_distributed()

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    apply_runtime_environment(config)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    dtype = getattr(torch, config["model"].get("dtype", "bfloat16"))

    if is_main_process():
        rank0_print(f"从 {config_path} 加载配置")
        print(json.dumps(config, indent=2, ensure_ascii=False))

    run = None
    if is_main_process():
        wandb_id = config["model"].get("wandb_id")
        if wandb_id is None:
            run = wandb.init(
                project=config["wandb"]["project"],
                name=config["wandb"]["run_name"],
                config=config,
            )
        else:
            run = wandb.init(
                project=config["wandb"]["project"],
                id=wandb_id,
                resume="must",
                config=config,
            )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["model_path"],
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompt_template = None
    prompt_path = config["data"].get("prompt_path")
    if prompt_path:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

    model_load_kwargs = build_model_load_kwargs(config)
    model_load_kwargs.pop("device_map", None)
    model_load_kwargs.pop("max_memory", None)
    policy = AutoModelForCausalLM.from_pretrained(config["model"]["model_path"], **model_load_kwargs)
    if config["training"].get("gradient_checkpointing", True):
        policy.gradient_checkpointing_enable()
    policy.config.use_cache = False
    policy.train()
    policy = wrap_model_with_fsdp(policy, config, local_rank)

    optimizer = AdamW(policy.parameters(), lr=float(config["training"]["learning_rate"]))
    n_grpo_steps = int(config["training"]["n_grpo_steps"])
    epochs_per_batch = int(config["training"]["epochs_per_rollout_batch"])
    warmup_ratio = float(config["training"].get("warmup_ratio", 0.0))
    global_step = int(config["model"].get("start_step") or 0)
    num_optimizer_steps = max(n_grpo_steps * epochs_per_batch, 1)
    warmup_steps = int(num_optimizer_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_optimizer_steps,
        last_epoch=(global_step * epochs_per_batch) - 1,
    )

    dataset = PromptAnswerDataset(
        config["data"]["train_path"],
        prompt_template=prompt_template,
        max_samples=config["data"].get("max_samples"),
    )
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    questions_per_batch = int(config["training"]["questions_per_rank"])
    dataloader = DataLoader(
        dataset,
        batch_size=questions_per_batch,
        sampler=sampler,
        collate_fn=collate_prompt_batch,
        drop_last=True,
    )

    reward_fn = build_reward_wrapper(config["training"].get("reward_style", "boxed_only"))
    stop_strings = config["training"].get("stop_strings", [])
    group_size = int(config["training"]["group_size"])
    inference_batch_size = int(config["training"].get("inference_batch_size", 1))
    micro_batch_size = int(config["training"]["micro_batch_size"])
    clip_range = float(config["training"]["clip_range"])
    normalize_by_std = bool(config["training"].get("normalize_by_std", True))
    remove_length_norm = bool(config["training"].get("remove_length_norm", False))
    fixed_norm_length = int(config["training"].get("fixed_norm_length") or config["data"].get("max_seq_length", 4096))
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    save_steps = int(config["training"].get("save_steps", 0))
    top_p = float(config["training"].get("top_p", 0.95))

    output_dir = config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    evaluation_cfg = config.get("evaluation", {})
    eval_enabled = bool(evaluation_cfg.get("enabled", False))
    eval_every_steps = int(evaluation_cfg.get("eval_every_steps", 0))
    eval_process = None
    eval_request_queue = None
    eval_response_queue = None
    pending_eval_step: int | None = None
    pending_eval_snapshot: Dict[str, Any] | None = None
    best_metric_name = "eval/reward_mean" if eval_enabled else "train/reward_mean"

    if eval_enabled and is_main_process():
        eval_prompts, eval_truths = build_eval_examples(config, prompt_template)
        eval_process, eval_request_queue, eval_response_queue = start_evaluation_worker(
            config=config,
            eval_prompts=eval_prompts,
            eval_truths=eval_truths,
            stop_strings=stop_strings,
        )
        rank0_print(
            "启动异步评估 worker: "
            f"device=cuda:{int(evaluation_cfg.get('eval_gpu_index', 0))}, "
            f"examples={len(eval_prompts)}, rollouts={int(evaluation_cfg.get('num_rollouts_per_example', 1))}"
        )

    rank0_print(
        f"开始 FSDP RL 训练: world_size={world_size}, "
        f"questions_per_rank={questions_per_batch}, group_size={group_size}"
    )

    pbar = tqdm(total=n_grpo_steps, desc=f"GRPO Steps [rank {get_rank()}]", disable=not is_main_process())
    pbar.update(global_step)

    best_reward = float(config["model"].get("best_reward", 0.0))
    data_iter = iter(dataloader)
    while global_step < n_grpo_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            sampler.set_epoch(global_step + 1)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        prompts = batch["prompt"]
        ground_truths = batch["ground_truth"]

        set_generation_mode(policy, enabled=True)
        policy.eval()
        encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        prompt_lengths = encoded.attention_mask.sum(dim=-1).tolist()

        with torch.no_grad():
            generation_output = generate_with_fsdp_full_params(
                policy,
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                do_sample=True,
                temperature=float(config["training"]["sampling_temperature"]),
                top_p=top_p,
                min_new_tokens=int(config["training"]["sampling_min_tokens"]),
                max_new_tokens=int(config["training"]["sampling_max_tokens"]),
                num_return_sequences=group_size,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                synced_gpus=get_world_size() > 1,
                return_dict_in_generate=True,
            )

        sequences = generation_output.sequences
        total_questions = len(prompts)
        all_prompts: List[str] = []
        all_responses: List[str] = []
        all_ground_truths: List[str] = []
        completion_lengths: List[int] = []

        for q_idx in range(total_questions):
            for sample_idx in range(group_size):
                seq_idx = q_idx * group_size + sample_idx
                generated_ids = sequences[seq_idx][prompt_lengths[q_idx]:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                processed_text = postprocess_generation(generated_text, stop_strings)

                all_prompts.append(prompts[q_idx])
                all_responses.append(processed_text)
                all_ground_truths.append(ground_truths[q_idx])
                if tokenizer.pad_token_id is not None:
                    completion_lengths.append(int((generated_ids != tokenizer.pad_token_id).sum().item()))
                else:
                    completion_lengths.append(len(generated_ids))

        set_generation_mode(policy, enabled=False)
        policy.train()

        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=all_responses,
            repeated_ground_truths=all_ground_truths,
            group_size=group_size,
            advantage_eps=float(config["training"]["advantage_eps"]),
            normalize_by_std=normalize_by_std,
            length_panelty=bool(config["training"].get("length_panelty", False)),
        )

        tokenized_batch = tokenize_prompt_and_output(
            prompt_strs=all_prompts,
            output_strs=all_responses,
            tokenizer=tokenizer,
            max_length=config["data"]["max_seq_length"],
        )
        input_ids = tokenized_batch["input_ids"].to(device)
        response_mask = tokenized_batch["response_mask"].to(device)
        labels = tokenized_batch["labels"].to(device)
        attention_mask = tokenized_batch["attention_mask"].to(device)
        advantages = advantages.to(device).unsqueeze(1)
        raw_rewards = raw_rewards.to(device).unsqueeze(1)

        old_log_probs_chunks = []
        total_entropy = 0.0
        total_entropy_tokens = 0.0
        policy.eval()
        with torch.no_grad():
            for start in range(0, len(input_ids), inference_batch_size):
                batch_input_ids = input_ids[start : start + inference_batch_size]
                batch_labels = labels[start : start + inference_batch_size]
                batch_attention_mask = attention_mask[start : start + inference_batch_size]

                with torch.amp.autocast(device_type=device.type, dtype=dtype):
                    log_probs_dict = get_response_log_probs(
                        policy,
                        batch_input_ids,
                        batch_attention_mask,
                        batch_labels,
                        return_token_entropy=True,
                    )
                old_log_probs_chunks.append(log_probs_dict["log_probs"].detach())

                if "token_entropy" in log_probs_dict:
                    batch_response_mask = response_mask[start : start + inference_batch_size]
                    entropy_sum = (log_probs_dict["token_entropy"] * batch_response_mask).sum().item()
                    token_count = batch_response_mask.sum().item()
                    total_entropy += entropy_sum
                    total_entropy_tokens += token_count
        policy.train()

        old_log_probs = torch.cat(old_log_probs_chunks, dim=0)
        train_dataset_len = len(input_ids)
        actual_accum_steps = math.ceil(train_dataset_len / micro_batch_size)
        epoch_metrics = {"loss": [], "clip_ratio": [], "approx_kl": []}
        for _ in range(epochs_per_batch):
            shuffle_indices = torch.randperm(train_dataset_len, device=device)
            epoch_loss = 0.0
            epoch_clip = 0.0
            epoch_kl = 0.0
            optimizer.zero_grad(set_to_none=True)
            for micro_idx, start in enumerate(range(0, train_dataset_len, micro_batch_size)):
                batch_indices = shuffle_indices[start : start + micro_batch_size]
                should_sync = micro_idx == actual_accum_steps - 1
                with maybe_no_sync(policy, not should_sync):
                    with torch.amp.autocast(device_type=device.type, dtype=dtype):
                        log_probs_dict = get_response_log_probs(
                            policy,
                            input_ids[batch_indices],
                            attention_mask[batch_indices],
                            labels[batch_indices],
                        )
                        loss, step_metrics = grpo_microbatch_train_step(
                            policy_log_probs=log_probs_dict["log_probs"],
                            response_mask=response_mask[batch_indices],
                            gradient_accumulation_steps=actual_accum_steps,
                            loss_type=config["training"]["loss_type"],
                            advantages=advantages[batch_indices],
                            old_log_probs=old_log_probs[batch_indices],
                            cliprange=clip_range,
                            remove_length_norm=remove_length_norm,
                            fixed_norm_length=fixed_norm_length,
                        )
                epoch_loss += loss.item()
                epoch_clip += step_metrics["clip_ratio"].item() / actual_accum_steps
                epoch_kl += step_metrics["approx_kl"].item() / actual_accum_steps

            grad_norm = policy.clip_grad_norm_(max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_metrics["loss"].append(epoch_loss)
            epoch_metrics["clip_ratio"].append(epoch_clip)
            epoch_metrics["approx_kl"].append(epoch_kl)

        mean_reward = reward_meta["mean_reward"]
        mean_entropy = total_entropy / max(total_entropy_tokens, 1.0)
        local_metrics = {
            "train/reward_mean": mean_reward,
            "train/reward_std": raw_rewards.std().item(),
            "train/format_rate": reward_meta["format_rate"],
            "train/loss": float(np.mean(epoch_metrics["loss"])),
            "train/clip_fraction": float(np.mean(epoch_metrics["clip_ratio"])),
            "train/approx_kl": float(np.mean(epoch_metrics["approx_kl"])),
            "train/lr": float(scheduler.get_last_lr()[0]),
            "train/grad_norm": float(grad_norm.item()),
            "train/completion_len_mean": float(np.mean(completion_lengths)),
            "train/completion_len_max": float(np.max(completion_lengths)),
            "train/mean_token_entropy": mean_entropy,
        }
        reduced_metrics = reduce_metrics(local_metrics, device)

        global_step += 1
        pbar.update(1)
        if is_main_process():
            pbar.set_postfix(reward=f"{reduced_metrics['train/reward_mean']:.4f}")
            if run is not None:
                wandb.log(
                    {
                        **reduced_metrics,
                        "train/global_step": global_step,
                    }
                )

        if is_main_process() and eval_response_queue is not None and pending_eval_step is not None:
            try:
                eval_result = eval_response_queue.get_nowait()
            except queue.Empty:
                eval_result = None

            if eval_result is not None:
                result_step = int(eval_result["step"])
                if eval_result["error"] is not None:
                    raise RuntimeError(f"评估 worker 在 step {result_step} 失败: {eval_result['error']}")

                eval_metrics = eval_result["metrics"] or {}
                eval_metrics["train/global_step"] = result_step
                rank0_print(
                    f"异步评估完成: step={result_step}, "
                    f"eval/reward_mean={eval_metrics.get('eval/reward_mean', float('nan')):.4f}"
                )
                if run is not None:
                    wandb.log(eval_metrics)

                current_reward = float(eval_metrics["eval/reward_mean"])
                if current_reward > best_reward:
                    best_reward = current_reward
                    best_save_path = os.path.join(output_dir, "checkpoint-best")
                    rank0_print(
                        f"最佳 {best_metric_name} 更新到 {best_reward:.4f}，"
                        f" 对应 step={result_step}，保存到 {best_save_path}"
                    )
                    unwrapped_model = get_wrapped_model(policy)
                    os.makedirs(best_save_path, exist_ok=True)
                    unwrapped_model.save_pretrained(best_save_path, state_dict=pending_eval_snapshot)
                    tokenizer.save_pretrained(best_save_path)
                    if run is not None:
                        wandb.log(
                            {
                                "eval/best_reward": best_reward,
                                "eval/best_step": result_step,
                                "train/global_step": global_step,
                            }
                        )

                pending_eval_step = None
                pending_eval_snapshot = None

        should_dispatch_eval = False
        if eval_enabled and eval_every_steps > 0 and global_step % eval_every_steps == 0:
            if is_main_process():
                should_dispatch_eval = pending_eval_step is None
            should_dispatch_eval = maybe_broadcast_bool(should_dispatch_eval, device)

            if should_dispatch_eval:
                eval_snapshot = gather_full_state_dict(policy)
                if is_main_process():
                    pending_eval_step = global_step
                    pending_eval_snapshot = eval_snapshot
                    eval_request_queue.put({"step": global_step, "state_dict": eval_snapshot})
                    rank0_print(f"已发送 step={global_step} 权重到异步评估 worker")
            elif is_main_process():
                rank0_print(f"跳过 step={global_step} 的评估: 上一次评估仍在进行中")

        if not eval_enabled and reduced_metrics["train/reward_mean"] > best_reward:
            best_reward = reduced_metrics["train/reward_mean"]
            best_save_path = os.path.join(output_dir, "checkpoint-best")
            rank0_print(f"最佳 {best_metric_name} 更新到 {best_reward:.4f}，保存到 {best_save_path}")
            save_fsdp_model(policy, tokenizer, best_save_path)

        if save_steps > 0 and global_step % save_steps == 0:
            save_path = os.path.join(output_dir, f"checkpoint-step-{global_step}")
            rank0_print(f"定期保存 checkpoint: {save_path}")
            save_fsdp_model(policy, tokenizer, save_path)

    if is_main_process() and eval_response_queue is not None and pending_eval_step is not None:
        rank0_print(f"等待最后一次异步评估完成: step={pending_eval_step}")
        eval_result = eval_response_queue.get()
        if eval_result["error"] is not None:
            raise RuntimeError(f"评估 worker 在 step {eval_result['step']} 失败: {eval_result['error']}")
        eval_metrics = eval_result["metrics"] or {}
        eval_metrics["train/global_step"] = int(eval_result["step"])
        if run is not None:
            wandb.log(eval_metrics)
        current_reward = float(eval_metrics["eval/reward_mean"])
        if current_reward > best_reward:
            best_reward = current_reward
            best_save_path = os.path.join(output_dir, "checkpoint-best")
            rank0_print(
                f"最终最佳 {best_metric_name} 更新到 {best_reward:.4f}，"
                f" 对应 step={int(eval_result['step'])}，保存到 {best_save_path}"
            )
            unwrapped_model = get_wrapped_model(policy)
            os.makedirs(best_save_path, exist_ok=True)
            unwrapped_model.save_pretrained(best_save_path, state_dict=pending_eval_snapshot)
            tokenizer.save_pretrained(best_save_path)

    final_path = os.path.join(output_dir, "checkpoint-final")
    rank0_print(f"训练完成，保存最终 checkpoint: {final_path}")
    save_fsdp_model(policy, tokenizer, final_path)

    if is_main_process() and eval_request_queue is not None:
        eval_request_queue.put(None)
        eval_process.join(timeout=10)

    if run is not None:
        run.finish()
    destroy_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/grpo_fsdp_qwen3_config.yaml")
    args = parser.parse_args()
    train(args.config)
