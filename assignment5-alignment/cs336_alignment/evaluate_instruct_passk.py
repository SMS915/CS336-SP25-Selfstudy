import os
import sys
import json
import time
import yaml
import argparse
import numpy as np
from vllm import LLM, SamplingParams
from typing import List, Dict, Callable, Any
from transformers.models.auto.tokenization_auto import AutoTokenizer

from cs336_alignment.device_config import apply_runtime_environment, build_runtime_wrapper_from_flat_config
from cs336_alignment.utils import format_prompt_for_instruct
from cs336_alignment.drgrpo_grader import qwen_instruct_reward_fn

def load_data(file_path: str, max_samples: int = 0) -> List[Dict]:
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples > 0 and i >= max_samples:
                break
            examples.append(json.loads(line))
    print(f"已加载数据: {len(examples)} 条")
    return examples

def formatting_prompt_qwen(examples: List[Dict], tokenizer: Any) -> List[str]:
    """使用官方模板构建适合 Instruct 模型的 Prompt"""
    prompts = []
    for ex in examples:
        prompt = format_prompt_for_instruct(ex["problem"], tokenizer)
        prompts.append(prompt)
    return prompts


def validate_model_path_for_vllm(model_path: str) -> None:
    if not os.path.isdir(model_path):
        return

    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    model_config_path = os.path.join(model_path, "config.json")
    if os.path.exists(adapter_config_path) and not os.path.exists(model_config_path):
        raise ValueError(
            "当前 model_path 指向的是 LoRA adapter 目录，而不是完整模型目录。"
            "这个目录只有 adapter_config.json / adapter_model.safetensors，"
            "vLLM 无法直接按完整模型加载。"
            "请先运行 `python cs336_alignment/merge_lora_adapter.py --adapter_path "
            f"{model_path} --output_path <merged_dir>` 合并权重，再把 model_path 改成合并后的目录。"
        )

def evaluate_vllm_pass_k(
    vllm_model: LLM, 
    reward_fn: Callable[[str, str], dict], 
    prompts: List[str], 
    examples: List[Dict], 
    eval_sampling_params: SamplingParams,
    pass_k: int = 1,
) -> List[Dict]:

    total_samples = len(prompts)
    final_results = [None] * total_samples
    pending_indices = list(range(total_samples))
    pass_m_history = []
    solved_indices = set()
    start_time = time.time()

    for attempt in range(1, pass_k + 1):
        if not pending_indices: 
            print(f"所有题目已在第 {attempt-1} 轮全部解决，正在填充后续数据...")
            # 将剩余的轮次全部填为 1.0
            while len(pass_m_history) < pass_k:
                pass_m_history.append(1.0)
            break
        print(f"=== 尝试第 {attempt}/{pass_k} 轮 (剩余 {len(pending_indices)} 题) ===")
        current_prompts = [prompts[i] for i in pending_indices]
        outputs = vllm_model.generate(current_prompts, eval_sampling_params, use_tqdm=True)
        
        next_pending_indices = []
        for idx_in_pending, output in enumerate(outputs):
            original_idx = pending_indices[idx_in_pending]
            generated_text = output.outputs[0].text
            example = examples[original_idx]
            truth = example.get("answer") or example.get("solution") or ""

            metrics = reward_fn(generated_text, truth)
            result_entry = {
                "problem": example["problem"],
                "gold_solution": truth,
                "generated_text": generated_text,
                "metrics": metrics,
                "attempt_id": attempt
            }
            final_results[original_idx] = result_entry
            
            if metrics.get("reward", 0.0) == 1.0:
                solved_indices.add(original_idx)
            elif attempt < pass_k:
                next_pending_indices.append(original_idx)

        pending_indices = next_pending_indices
        pass_m_history.append(len(solved_indices) / total_samples)

        print(f">>> 累计解决率 (Pass@{attempt}): {len(solved_indices) / total_samples:.2%}")

    print("\n" + "="*20)
    print(f"Pass@1 to Pass@{pass_k} 趋势数据:")
    print("[", end='')
    for i, acc in enumerate(pass_m_history):
        print(f"{acc:.3f}", end=', ') if i != len(pass_m_history) - 1 else print(f"{acc:.3f}", end="")
    print("]")
    print("="*20 + "\n")

    end_time = time.time()
    print(f"评估完成，Pass@{pass_k} 总耗时: {end_time - start_time:.2f}秒")

    correct_count = 0
    ans_error_count = 0
    format_error_count = 0
    total_len = 0
    
    # 此时 final_results 中存储的是每道题最好的结果（如果做对）或最后的结果（如果全错）
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
    
    print("\n" + "="*40)
    print(f"Pass@{pass_k} 最终评估结果:")
    print(f"样本总数: {total_samples}")
    print(f"解决率 (至少一次做对): {accuracy: .2%}")
    print(f"格式正确但答案错误 (最终状态): {ans_error_count / total_samples:.2%}")
    print(f"格式错误 (最终状态): {format_error_count / total_samples:.2%}")
    print(f"平均生成的字符长度: {avg_len:.2f}")
    print("="*40 + "\n")

    return final_results

def run_evaluate(config: Dict[str, Any]):
    apply_runtime_environment(build_runtime_wrapper_from_flat_config(config))
    max_samples = config.get('max_samples', 0)
    examples = load_data(config['example_path'], max_samples)
    validate_model_path_for_vllm(config["model_path"])
    
    # 加载 Tokenizer用于 Prompt 模板渲染
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=True)

    test_prompt = format_prompt_for_instruct("1+1=?", tokenizer)
    print(f"--- Prompt Sample ---\n{test_prompt}\n---------------------")
    
    # 格式化 Prompt
    formatted_input = formatting_prompt_qwen(examples=examples, tokenizer=tokenizer)
    
    # 初始化 vLLM
    llm = LLM(
        model=config['model_path'], 
        dtype=config.get('dtype', 'bfloat16'), 
        tensor_parallel_size=config.get('tensor_parallel_size', 1),
        gpu_memory_utilization=config.get('gpu_memory_utilization', 0.90),
        trust_remote_code=True,
    )

    # Qwen-Math-Instruct 官方推荐停止符
    stop_tokens = ["<|im_end|>", "<|endoftext|>", "\n\n\n"] 
    if config.get('use_r1_format', False):
        stop_tokens.append("</answer>")

    eval_params = SamplingParams(
        temperature=config.get('temperature', 0.7),
        top_p=config.get('top_p', 0.9), 
        max_tokens=config.get('max_tokens', 2048),
        stop=stop_tokens,
        include_stop_str_in_output=True,
        n=1
    )

    # 执行评估
    eval_results = evaluate_vllm_pass_k(
        vllm_model=llm, 
        reward_fn=qwen_instruct_reward_fn, 
        prompts=formatted_input, 
        examples=examples, 
        eval_sampling_params=eval_params,
        pass_k=config.get('pass_k', 1)
    )

    # 保存结果
    with open(config['output_path'], 'w') as f:
        for res in eval_results:
            f.write(json.dumps(res) + "\n")

def parse_arguments() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Evaluate vLLM model performance with Pass@K.")

    # 配置文件
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config.')

    # 核心参数
    parser.add_argument('--example_path', type=str, help='Evaluation dataset (jsonl).')
    parser.add_argument('--output_path', type=str, help='Save path.')
    parser.add_argument('--model_path', type=str, help='Model checkpoint.')
    
    # 生成参数
    parser.add_argument('--max_tokens', type=int, help='Max tokens (truncation).')
    parser.add_argument('--temperature', type=float, help='Sampling temperature.')
    parser.add_argument('--top_p', type=float, help='Top-p sampling.')
    parser.add_argument('--dtype', type=str, help='vLLM dtype, e.g. bfloat16.')
    parser.add_argument('--tensor_parallel_size', type=int, help='Number of GPUs used by vLLM tensor parallel.')
    parser.add_argument('--gpu_memory_utilization', type=float, help='vLLM gpu memory utilization.')
    parser.add_argument('--cuda_visible_devices', type=str, help='Visible GPU ids, e.g. 0 or 0,1.')
    
    # Pass@K 参数
    parser.add_argument('--max_samples', type=int, help='Sampling temperature.')
    parser.add_argument('--pass_k', type=int, help='Number of attempts per problem (Pass@K).')

    args = parser.parse_args()

    final_config = {}
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config not found: {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                final_config.update(yaml_config)

    for key, value in vars(args).items():
        if key != 'config' and value is not None:
            final_config[key] = value
            
    # 设置默认值
    if 'pass_k' not in final_config:
        final_config['pass_k'] = 1
    if 'max_tokens' not in final_config:
        final_config['max_tokens'] = 1024
    if 'temperature' not in final_config:
        final_config['temperature'] = 1.0
    if 'dtype' not in final_config:
        final_config['dtype'] = 'bfloat16'

    return final_config

def validate_config(config: Dict[str, Any]) -> None:
    required_keys = ['example_path', 'output_path', 'model_path']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        print(f"Error: Missing required keys: {missing_keys}")
        sys.exit(1)

if __name__ == '__main__':
    config = parse_arguments()
    validate_config(config)

    print("=== 运行配置 ===")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("================\n")

    output_dir = os.path.dirname(config['output_path'])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    run_evaluate(config)
