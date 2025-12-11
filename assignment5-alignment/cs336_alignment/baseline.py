import os
import json
import time
from vllm import LLM, SamplingParams
from typing import List, Tuple, Dict, Callable
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def load_data(file_path: str) -> List[Dict]:
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def formatting_prompt(examples: List[Dict], prompt_template: str) -> List[str]:
    prompts = []
    for ex in examples:
        prompt = prompt_template.replace("{question}", ex["problem"])
        prompts.append(prompt)

    return prompts

def evaluate_vllm(vllm_model: LLM, reward_fn: Callable[[str, str], dict[str, float]], prompts: List[str], examples: List[Dict], eval_sampling_params: SamplingParams) -> List[Dict]:
    """
    使用给定的奖励函数评估 vLLM 模型在特定提示集上的表现。
    执行批量推理，计算生成结果与标准答案的匹配度，并统计准确率和格式错误率。

    Args:
        vllm_model (LLM): 已初始化的 vLLM 模型实例，用于执行推理。
        reward_fn (Callable[[str, str], dict[str, float]]): 评估函数。
            接收两个参数：(generated_text, truth)。
            返回一个字典，需包含以下键以进行分类统计：
            - "reward": float, 1.0 表示完全正确。
            - "format_reward": float, 1.0 表示格式正确（但答案可能错误）。
        prompts (List[str]): 输入给模型的 prompt 列表。
        examples (List[Dict]): 包含问题和标准答案的样本列表，需与 prompts 索引一一对应。
            每个字典必须包含 "solution" (标准答案) 和 "problem" (原始问题) 字段。
        eval_sampling_params (SamplingParams): vLLM 的生成采样参数（如 temperature, top_p 等）。

    Returns:
        List[Dict]: 包含每个样本详细评估结果的列表。列表中的每个字典结构如下：
            {
                "problem": str,          # 原始问题
                "gold_solution": str,    # 标准答案
                "generated_text": str,   # 模型生成的完整文本
                "metrics": dict          # reward_fn 返回的详细指标
            }
    """
    print(f"开始生成{len(prompts)}条数据")
    start_time = time.time()
    # vllm_model.generate 批处理，比逐条循环生成效率更高
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    end_time = time.time()
    print(f"生成完成，共用时{end_time - start_time}秒")
    results = []
    correct_count = 0      # 完全正确
    ans_error_count = 0    # 格式正确但答案错误
    format_error_count = 0 # 格式错误
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        example = examples[i]
        truth = example["solution"]
        metrics = reward_fn(generated_text, truth)
        if metrics.get("reward", 0.0) == 1.0:
            correct_count += 1
        elif metrics.get("format_reward", 0.0) == 1.0:
            ans_error_count += 1
        else:
            format_error_count += 1

        result_entry = {
            "problem": example["problem"],
            "gold_solution": example["solution"],
            "generated_text": generated_text,
            "metrics": metrics
        }

        results.append(result_entry)
    accuracy = correct_count / len(prompts)
    print("评估结果如下:")
    print(f"完全正确: {accuracy: .2%}")
    print(f"格式正确，答案错误: {ans_error_count / len(prompts):.2%}")
    print(f"格式错误: {format_error_count / len(prompts):.2%}")
    return results

def run_evaluate(example_path: str, prompt_path: str, output_path: str, model_path: str):
    """
    执行端到端的模型评估流程：加载数据、初始化模型、生成并保存结果。

    该函数负责协调完整的评估管道：读取原始数据和 Prompt 模板，配置 vLLM 模型环境，
    执行推理与奖励评分，最终将详细的评估日志持久化存储。

    Args:
        example_path (str): 测试数据集的文件路径。
            预期格式应能被内部的 `load_data` 函数解析。
        prompt_path (str): 包含 Prompt 模板的纯文本文件路径。
            文件内容将作为模板用于格式化输入数据。
        output_path (str): 评估结果的保存路径。
            结果将以 JSONL (每行一个 JSON 对象) 格式写入。
        model_path (str): 模型路径或 HuggingFace 模型 ID。

    Returns:
        None: 此函数不返回数据，结果直接写入 `output_path` 指定的文件。

    Note:
        函数内部使用了硬编码的模型配置：
        - dtype="bfloat16"
        - gpu_memory_utilization=0.9
        - temperature=1.0, top_p=1.0 (贪婪/确定性采样通常应设 temp=0，此处设为1.0可能是为了多样性或特定需求)
        - stop_tokens=["</answer>"]
    """
    examples = load_data(example_path)
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()
    formatted_input = formatting_prompt(examples=examples, prompt_template=prompt_template)

    llm = LLM(model = model_path, dtype="bfloat16", gpu_memory_utilization = 0.9, trust_remote_code = True)

    eval_params = SamplingParams(temperature = 1.0, top_p = 1.0, max_tokens=1024, stop=["</answer>"],
                               include_stop_str_in_output=True)

    eval_results = evaluate_vllm(vllm_model=llm, reward_fn=r1_zero_reward_fn, prompts=formatted_input, examples=examples, eval_sampling_params=eval_params)

    with open(output_path, 'w') as f:
        for res in eval_results:
            f.write(json.dumps(res) + "\n")
    print(f" 结果已保存至: {output_path}")

if __name__ == '__main__':
    EXAMPLE_PATH = 'data/MATH/validation.jsonl'
    PROMPT_PATH = 'cs336_alignment/prompts/r1_zero.prompt'
    OUTPUT_PATH = 'results/sft_v1_result.jsonl'
    MODEL_PATH = 'checkpoints/sft_v1'
    os.makedirs('results', exist_ok=True)
    run_evaluate(EXAMPLE_PATH, PROMPT_PATH, OUTPUT_PATH, MODEL_PATH)





