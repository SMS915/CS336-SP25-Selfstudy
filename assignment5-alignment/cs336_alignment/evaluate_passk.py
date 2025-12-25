import os
import sys
import json
import time
import yaml
import argparse
import numpy as np
from vllm import LLM, SamplingParams
from typing import List, Dict, Callable, Any
# 假设 utils 里有 robust_reward_fn，如果没有请替换为你的实际 reward 函数
from cs336_alignment.utils import robust_reward_fn

def load_data(file_path: str, max_samples: int = 0) -> List[Dict]:
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # 如果 max_samples > 0 且当前索引已达到限制，则停止读取
            if max_samples > 0 and i >= max_samples:
                break
            examples.append(json.loads(line))

    print(f"已加载数据: {len(examples)} 条 (限制: {max_samples if max_samples > 0 else '全部'})")
    return examples

def formatting_prompt(examples: List[Dict], prompt_template: str) -> List[str]:
    prompts = []
    for ex in examples:
        prompt = prompt_template.replace("{question}", ex["problem"])
        prompts.append(prompt)
    return prompts

def evaluate_vllm_pass_k(
    vllm_model: LLM, 
    reward_fn: Callable[[str, str], dict], 
    prompts: List[str], 
    examples: List[Dict], 
    eval_sampling_params: SamplingParams,
    pass_k: int = 1
) -> List[Dict]:
    """
    实现了 Pass@K 的评估逻辑，并带有 Early Stopping（早停）机制。
    如果某道题在第 i 次尝试做对了，就不会进行第 i+1 次生成，节省算力。
    """
    total_samples = len(prompts)
    print(f"开始 Pass@{pass_k} 评估，共 {total_samples} 条数据")
    
    # 初始化结果列表，长度与样本一致，初始为 None
    # 最终这个列表里存的将是：做对的那一次生成的 result，或者（如果K次都错）最后一次错误的 result
    final_results = [None] * total_samples
    
    # 记录目前还需要生成的样本索引
    # 初始状态：所有题目都需要跑
    pending_indices = list(range(total_samples))
    pass_m_history = []
    
    start_time = time.time()
    
    # 循环 K 次
    for attempt in range(1, pass_k + 1):
        if not pending_indices:
            print("所有题目已在之前的尝试中解决，提前结束评估。")
            while len(pass_m_history) < pass_k:
                pass_m_history.append(1.0)
            break
            
        print(f"=== 尝试第 {attempt}/{pass_k} 轮 (剩余 {len(pending_indices)} 题) ===")
        
        # 1. 准备当前轮次的 Prompts
        current_prompts = [prompts[i] for i in pending_indices]
        
        # 2. 批量生成 (vLLM 内部会自动批处理)
        # 注意：这里我们每次只生成 1 个 (n=1)，靠外层循环来实现 K
        outputs = vllm_model.generate(current_prompts, eval_sampling_params, use_tqdm=True)
        
        # 下一轮需要跑的索引列表
        next_pending_indices = []
        
        # 3. 检查结果
        for idx_in_pending, output in enumerate(outputs):
            original_idx = pending_indices[idx_in_pending] # 对应原始数据的索引
            
            generated_text = output.outputs[0].text
            example = examples[original_idx]
            truth = example.get("answer") or example.get("solution")
            if truth is None:
                print(f"{original_idx} has no answer or solution")
                truth = ""
            assert isinstance(truth, str)
            # 调用 Reward 函数评分
            metrics = reward_fn(generated_text, truth)
            
            result_entry = {
                "problem": example["problem"],
                "gold_solution": truth,
                "generated_text": generated_text,
                "metrics": metrics,
                "attempt_id": attempt # 记录是在第几次尝试做出来的
            }
            
            # 更新最终结果 (无论对错先存进去，如果是错的，可能会被下一轮覆盖)
            final_results[original_idx] = result_entry
            
            # 判断是否做对 (Reward == 1.0)
            if metrics.get("reward", 0.0) == 1.0:
                # 做对了！不需要进入下一轮 pending 列表
                pass 
            else:
                # 没做对，如果还有剩余次数，加入下一轮
                if attempt < pass_k:
                    next_pending_indices.append(original_idx)
        
        # 更新待处理列表
        pending_indices = next_pending_indices
        solved_count = total_samples - len(pending_indices)
        current_acc = solved_count / total_samples
        pass_m_history.append(current_acc)
        print(f">>> 累计解决率 (Pass@{attempt}): {current_acc:.2%}")
    
    print("\n" + "="*20)
    print(f"Pass@1 to Pass@{pass_k} 趋势数据:")
    print(pass_m_history)
    print("="*20 + "\n")

    end_time = time.time()
    print(f"评估完成，Pass@{pass_k} 总耗时: {end_time - start_time:.2f}秒")

    # --- 统计最终结果 ---
    correct_count = 0
    ans_error_count = 0
    format_error_count = 0
    total_len = 0
    
    # 此时 final_results 中存储的是每道题最好的结果（如果做对）或最后的结果（如果全错）
    for res in final_results:
        # 计算长度 (包含 Prompt 的长度在 vLLM output 里不好直接减，这里粗略计算 generated_text 长度)
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
    """
    Args:
        config (Dict): 包含所有参数的字典
    """
    max_samples = config.get('max_samples', 0)
    examples = load_data(config['example_path'], max_samples)
    with open(config['prompt_path'], 'r') as f:
        prompt_template = f.read()
    
    formatted_input = formatting_prompt(examples=examples, prompt_template=prompt_template)
    
    # vLLM 初始化
    llm = LLM(
        model=config['model_path'], 
        dtype="bfloat16", 
        gpu_memory_utilization=0.95, 
        trust_remote_code=True,
        # max_model_len=8192 # 如果遇到显存不够可以开启
    )

    # 采样参数配置
    # 注意：这里 max_tokens 起到了“长度截断”的作用
    # 如果模型生成超过这个长度，vLLM 会强制停止，且 finish_reason 为 length
    eval_params = SamplingParams(
        temperature=config.get('temperature', 1.0), 
        top_p=config.get('top_p', 1.0), 
        max_tokens=config.get('max_tokens', 1024),
        stop=["</answer>"], # 遇到这个标签停止
        include_stop_str_in_output=True,
        n=1 # 每次只生成一个，通过外层循环控制 Pass@K
    )
    
    print(f"最大输出 Token 限制 (截断): {config.get('max_tokens', 1024)}")
    print(f"Temperature: {config.get('temperature', 1.0)}")
    print(f"Pass K: {config.get('pass_k', 1)}")

    # 执行 Pass@K 评估
    eval_results = evaluate_vllm_pass_k(
        vllm_model=llm, 
        reward_fn=robust_reward_fn, 
        prompts=formatted_input, 
        examples=examples, 
        eval_sampling_params=eval_params,
        pass_k=config.get('pass_k', 1)
    )

    with open(config['output_path'], 'w') as f:
        for res in eval_results:
            f.write(json.dumps(res) + "\n")
    print(f"结果已保存至: {config['output_path']}")

def parse_arguments() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Evaluate vLLM model performance with Pass@K.")

    # 配置文件
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config.')

    # 核心参数
    parser.add_argument('--example_path', type=str, help='Evaluation dataset (jsonl).')
    parser.add_argument('--prompt_path', type=str, help='Prompt template file.')
    parser.add_argument('--output_path', type=str, help='Save path.')
    parser.add_argument('--model_path', type=str, help='Model checkpoint.')
    
    # 生成参数
    parser.add_argument('--max_tokens', type=int, help='Max tokens (truncation).')
    parser.add_argument('--temperature', type=float, help='Sampling temperature.')
    parser.add_argument('--top_p', type=float, help='Top-p sampling.')
    
    # 新增 Pass@K 参数
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

    return final_config

def validate_config(config: Dict[str, Any]) -> None:
    required_keys = ['example_path', 'prompt_path', 'output_path', 'model_path']
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