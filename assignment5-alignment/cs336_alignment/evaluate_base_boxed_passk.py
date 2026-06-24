import os
import sys
import json
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List

import yaml
from tqdm import tqdm


def _bootstrap_cuda_visible_devices() -> None:
    cli_cuda_visible_devices = None
    config_path = None

    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--cuda_visible_devices" and i + 1 < len(argv):
            cli_cuda_visible_devices = argv[i + 1]
        elif arg.startswith("--cuda_visible_devices="):
            cli_cuda_visible_devices = arg.split("=", 1)[1]
        elif arg == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]
        elif arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]

    config_cuda_visible_devices = None
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
            config_cuda_visible_devices = config_data.get("cuda_visible_devices")
        except Exception:
            config_cuda_visible_devices = None

    cuda_visible_devices = cli_cuda_visible_devices or config_cuda_visible_devices
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
        print(
            f"Bootstrap CUDA_VISIBLE_DEVICES={cuda_visible_devices} "
            "(注意: 程序里的逻辑 GPU 0 会映射到这里指定的第一张物理卡)",
            file=sys.stderr,
        )


_bootstrap_cuda_visible_devices()

import argparse
from vllm import LLM, SamplingParams

from cs336_alignment.device_config import (
    apply_runtime_environment,
    build_runtime_wrapper_from_flat_config,
    get_vllm_load_kwargs,
)
from cs336_alignment.drgrpo_grader import question_only_reward_fn


DEFAULT_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer within \\boxed{}.\n\n"
    "Problem:\n"
    "{question}\n\n"
    "Solution:\n"
)


def load_data(file_path: str, max_samples: int = 0) -> List[Dict]:
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples > 0 and i >= max_samples:
                break
            examples.append(json.loads(line))
    print(f"已加载数据: {len(examples)} 条")
    return examples


def load_prompt_template(prompt_path: str | None) -> str:
    if prompt_path is None:
        return DEFAULT_PROMPT_TEMPLATE

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def format_prompts(examples: List[Dict], prompt_template: str) -> List[str]:
    prompts = []
    for ex in examples:
        prompts.append(prompt_template.replace("{question}", ex["problem"]))
    return prompts


def log_requested_physical_gpu_status(cuda_visible_devices: str | None) -> None:
    if not cuda_visible_devices or shutil.which("nvidia-smi") is None:
        return

    requested_gpu_ids = []
    for raw_gpu_id in str(cuda_visible_devices).split(","):
        gpu_id = raw_gpu_id.strip()
        if gpu_id.isdigit():
            requested_gpu_ids.append(int(gpu_id))

    if not requested_gpu_ids:
        return

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        print(f"读取 GPU 显存状态失败，跳过预检查: {exc}")
        return

    gpu_memory_map: Dict[int, tuple[int, int]] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpu_memory_map[int(parts[0])] = (int(parts[1]), int(parts[2]))
        except ValueError:
            continue

    for logical_idx, physical_idx in enumerate(requested_gpu_ids):
        if physical_idx not in gpu_memory_map:
            print(f"预检查: 未找到物理 GPU {physical_idx} 的显存信息。")
            continue
        used_mib, free_mib = gpu_memory_map[physical_idx]
        print(
            f"预检查: 逻辑 GPU {logical_idx} -> 物理 GPU {physical_idx}, "
            f"已用 {used_mib} MiB, 剩余 {free_mib} MiB"
        )
        if free_mib < 14000:
            print(
                f"警告: 物理 GPU {physical_idx} 当前空闲显存偏少，"
                "vLLM 初始化或生成时可能 OOM。"
            )


def evaluate_vllm_pass_k(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    examples: List[Dict],
    eval_sampling_params: SamplingParams,
    pass_k: int = 1,
    prompt_batch_size: int = 8,
) -> List[Dict]:
    total_samples = len(prompts)
    final_results = [None] * total_samples
    pending_indices = list(range(total_samples))
    pass_m_history = []
    solved_indices = set()
    start_time = time.time()

    for attempt in range(1, pass_k + 1):
        if not pending_indices:
            print(f"所有题目已在第 {attempt - 1} 轮全部解决，正在填充后续数据...")
            while len(pass_m_history) < pass_k:
                pass_m_history.append(1.0)
            break

        next_pending_indices = []
        print(
            f"=== 尝试第 {attempt}/{pass_k} 轮 (剩余 {len(pending_indices)} 题, "
            f"prompt_batch_size={prompt_batch_size}) ==="
        )

        chunk_starts = range(0, len(pending_indices), prompt_batch_size)
        for start in tqdm(chunk_starts, desc=f"Attempt {attempt}/{pass_k}", unit="chunk"):
            batch_pending_indices = pending_indices[start : start + prompt_batch_size]
            current_prompts = [prompts[i] for i in batch_pending_indices]
            outputs = vllm_model.generate(current_prompts, eval_sampling_params, use_tqdm=False)

            for idx_in_pending, output in enumerate(outputs):
                original_idx = batch_pending_indices[idx_in_pending]
                generated_text = output.outputs[0].text
                example = examples[original_idx]
                truth = example.get("answer") or example.get("solution") or ""

                metrics = reward_fn(generated_text, truth)
                result_entry = {
                    "problem": example["problem"],
                    "gold_solution": truth,
                    "generated_text": generated_text,
                    "metrics": metrics,
                    "attempt_id": attempt,
                }
                final_results[original_idx] = result_entry

                if metrics.get("reward", 0.0) == 1.0:
                    solved_indices.add(original_idx)
                elif attempt < pass_k:
                    next_pending_indices.append(original_idx)

        pending_indices = next_pending_indices
        pass_m_history.append(len(solved_indices) / total_samples)
        print(f">>> 累计解决率 (Pass@{attempt}): {len(solved_indices) / total_samples:.2%}")

    print("\n" + "=" * 20)
    print(f"Pass@1 to Pass@{pass_k} 趋势数据:")
    print("[", end="")
    for i, acc in enumerate(pass_m_history):
        if i != len(pass_m_history) - 1:
            print(f"{acc:.3f}", end=", ")
        else:
            print(f"{acc:.3f}", end="")
    print("]")
    print("=" * 20 + "\n")

    end_time = time.time()
    print(f"评估完成，Pass@{pass_k} 总耗时: {end_time - start_time:.2f}秒")

    correct_count = 0
    ans_error_count = 0
    format_error_count = 0
    total_len = 0

    for res in final_results:
        total_len += len(res["generated_text"])

        metrics = res["metrics"]
        if metrics.get("reward", 0.0) == 1.0:
            correct_count += 1
        elif metrics.get("format_reward", 0.0) == 1.0:
            ans_error_count += 1
        else:
            format_error_count += 1

    accuracy = correct_count / total_samples
    avg_len = total_len / total_samples

    print("\n" + "=" * 40)
    print(f"Pass@{pass_k} 最终评估结果:")
    print(f"样本总数: {total_samples}")
    print(f"解决率 (至少一次做对): {accuracy:.2%}")
    print(f"抽取到 boxed 但答案错误 (最终状态): {ans_error_count / total_samples:.2%}")
    print(f"未抽取到 boxed (最终状态): {format_error_count / total_samples:.2%}")
    print(f"平均生成的字符长度: {avg_len:.2f}")
    print("=" * 40 + "\n")

    return final_results


def run_evaluate(config: Dict[str, Any]):
    runtime_wrapper = build_runtime_wrapper_from_flat_config(config)
    apply_runtime_environment(runtime_wrapper)
    max_samples = config.get("max_samples", 0)
    examples = load_data(config["example_path"], max_samples)
    if not examples:
        raise ValueError("评测集为空，无法执行评测。")
    prompt_template = load_prompt_template(config.get("prompt_path"))
    formatted_input = format_prompts(examples=examples, prompt_template=prompt_template)

    print(f"--- Prompt Sample ---\n{formatted_input[0]}\n---------------------")

    prompt_batch_size = int(config.get("prompt_batch_size", 8))
    if prompt_batch_size <= 0:
        raise ValueError("prompt_batch_size 必须为正整数。")
    log_requested_physical_gpu_status(config.get("cuda_visible_devices"))

    llm_kwargs = {
        "model": config["model_path"],
        **get_vllm_load_kwargs(
            runtime_wrapper,
            default_gpu_memory_utilization=config.get("gpu_memory_utilization", 0.8),
        ),
    }
    if config.get("max_num_seqs") is not None:
        llm_kwargs["max_num_seqs"] = int(config["max_num_seqs"])
    else:
        llm_kwargs["max_num_seqs"] = prompt_batch_size
        print(f"未显式设置 max_num_seqs，默认使用 prompt_batch_size={prompt_batch_size}。")
    llm = LLM(**llm_kwargs)

    temperature = config.get("temperature", 0.0)
    if config.get("pass_k", 1) > 1 and temperature == 0.0:
        print("警告: 当前 temperature=0.0，Pass@K 的多轮生成会几乎完全一致。")
    if temperature == 0.0 and config.get("top_p", 1.0) != 1.0:
        print("检测到 greedy 评测，自动将 top_p 设为 1.0 以减少不必要的采样开销。")

    eval_params = SamplingParams(
        temperature=temperature,
        top_p=1.0 if temperature == 0.0 else config.get("top_p", 1.0),
        max_tokens=config.get("max_tokens", 1024),
        n=1,
    )

    eval_results = evaluate_vllm_pass_k(
        vllm_model=llm,
        reward_fn=question_only_reward_fn,
        prompts=formatted_input,
        examples=examples,
        eval_sampling_params=eval_params,
        pass_k=config.get("pass_k", 1),
        prompt_batch_size=prompt_batch_size,
    )

    with open(config["output_path"], "w", encoding="utf-8") as f:
        for res in eval_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"结果已保存至: {config['output_path']}")


def parse_arguments() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Evaluate base models with boxed-answer-only Pass@K.")

    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--example_path", type=str, help="Evaluation dataset (jsonl).")
    parser.add_argument("--prompt_path", type=str, help="Optional prompt template path.")
    parser.add_argument("--output_path", type=str, help="Save path.")
    parser.add_argument("--model_path", type=str, help="Model checkpoint.")
    parser.add_argument("--max_tokens", type=int, help="Max generated tokens.")
    parser.add_argument("--temperature", type=float, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, help="Top-p sampling.")
    parser.add_argument("--gpu_memory_utilization", type=float, help="vLLM GPU memory utilization.")
    parser.add_argument("--dtype", type=str, help="vLLM dtype, e.g. bfloat16.")
    parser.add_argument("--tensor_parallel_size", type=int, help="Number of GPUs used by vLLM tensor parallel.")
    parser.add_argument("--cuda_visible_devices", type=str, help="Visible GPU ids, e.g. 0 or 0,1.")
    parser.add_argument("--prompt_batch_size", type=int, help="How many prompts to submit to vLLM per chunk.")
    parser.add_argument("--max_num_seqs", type=int, help="Optional vLLM max_num_seqs override.")
    parser.add_argument("--max_samples", type=int, help="Limit number of eval samples.")
    parser.add_argument("--pass_k", type=int, help="Number of attempts per problem.")

    args = parser.parse_args()

    final_config: Dict[str, Any] = {}
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config not found: {args.config}")
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                final_config.update(yaml_config)

    for key, value in vars(args).items():
        if key != "config" and value is not None:
            final_config[key] = value

    if "pass_k" not in final_config:
        final_config["pass_k"] = 1
    if "max_tokens" not in final_config:
        final_config["max_tokens"] = 1024
    if "temperature" not in final_config:
        final_config["temperature"] = 0.0
    if "top_p" not in final_config:
        final_config["top_p"] = 1.0
    if "dtype" not in final_config:
        final_config["dtype"] = "bfloat16"
    if "prompt_batch_size" not in final_config:
        final_config["prompt_batch_size"] = 8

    return final_config


def validate_config(config: Dict[str, Any]) -> None:
    required_keys = ["example_path", "output_path", "model_path"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        print(f"Error: Missing required keys: {missing_keys}")
        sys.exit(1)


if __name__ == "__main__":
    config = parse_arguments()
    validate_config(config)

    print("=== 运行配置 ===")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("================\n")

    output_dir = os.path.dirname(config["output_path"])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    run_evaluate(config)
