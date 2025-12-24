import os
import sys
import json
import time
import yaml
import argparse
from vllm import LLM, SamplingParams
from typing import List, Dict, Callable, Any
from cs336_alignment.utils import robust_reward_fn
from transformers.generation.configuration_utils import GenerationConfig

def load_data(file_path: str) -> List[Dict]:
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def formatting_prompt(examples: List[Dict], prompt_template: str) -> List[str]:
    """
    使用提供的模板批量格式化测试用例，生成最终的模型输入 Prompt。
    遍历示例列表，将每个示例中的问题文本注入到模板的指定占位符中。

    Args:
        examples (List[Dict]): 测试用例列表。
            列表中的每个字典必须包含 "problem" 键，代表原始问题文本。
        prompt_template (str): 用于构建prompt的字符串模板
            该字符串必须包含"{question}"占位符，以便替换为实际问题。

    Returns:
        List[str]: 格式化后的完整 Prompt 列表。
            长度与输入examples相同，并顺序一一对应

    Raises:
        KeyError: 如果 examples 中的字典缺少 "problem" 键，将抛出此异常。
    """
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
            - "answer_reward": float, 1.0 表示答案正确（实际场景下和reward没区别）
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
    avg_len = 0            # 记录回复的字符长度

    for i,(output, prompt) in enumerate(zip(outputs, prompts)):
        generated_text = output.outputs[0].text
        avg_len += (len(generated_text) - len(prompt)) / len(outputs)
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
    print(f"平均response字符长度: {avg_len:.2f}")
    return results

def run_evaluate(example_path: str, prompt_path: str, output_path: str, model_path: str,
                  max_tokens: int = 1024, top_p: float = 1.0, temperature: float = 1.0):
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
        根据官方handout, 函数内部使用了硬编码的模型配置：
        - dtype="bfloat16"
        - gpu_memory_utilization=0.9
        - stop_tokens=["</answer>"]
    """
    examples = load_data(example_path)
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()
    formatted_input = formatting_prompt(examples=examples, prompt_template=prompt_template)

    # rope_scaling_config = {"rope_type": "dynamic", "factor": 2.0}
    
    target_max_len = 8192

    llm = LLM(model = model_path, 
              dtype="bfloat16", 
              gpu_memory_utilization = 0.95, 
              trust_remote_code = True,
            #   rope_scaling=rope_scaling_config,
            #   max_model_len=target_max_len
              )

    eval_params = SamplingParams(temperature = temperature, top_p = top_p, max_tokens=max_tokens ,stop=["</answer>"],
                               include_stop_str_in_output=True)
    print(f"最大输出长度为{max_tokens}")

    eval_results = evaluate_vllm(vllm_model=llm, reward_fn=robust_reward_fn, prompts=formatted_input, examples=examples, eval_sampling_params=eval_params)

    with open(output_path, 'w') as f:
        for res in eval_results:
            f.write(json.dumps(res) + "\n")
    print(f" 结果已保存至: {output_path}")


def parse_arguments() -> Dict[str, Any]:
    """

    解析命令行参数并加载 YAML 配置文件（如果提供）。

    优先级逻辑：命令行参数 (CLI) > YAML 配置文件 > 默认值（如果有）。

    Returns:
        Dict[str, Any]: 包含最终运行配置的字典。
    """
    parser = argparse.ArgumentParser(description="Evaluate vLLM model performance.")

    # 配置文件参数
    parser.add_argument('--config', type=str, default=None, help='Path to the YAML configuration file.')

    # 具体的运行参数 (默认值为 None，以便区分是否在命令行中指定了该参数)
    parser.add_argument('--example_path', type=str, help='Path to the evaluation dataset (jsonl).')
    parser.add_argument('--prompt_path', type=str, help='Path to the prompt template file.')
    parser.add_argument('--output_path', type=str, help='Path to save evaluation results.')
    parser.add_argument('--model_path', type=str, help='Path to the model checkpoint or HF ID.')
    parser.add_argument('--max_tokens', type=int, help='Max token limitation for generate output')

    args = parser.parse_args()

    # 初始化最终配置字典
    final_config = {}

    # 如果提供了 YAML 配置文件，先加载它
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                final_config.update(yaml_config)

    # 使用命令行参数覆盖 YAML 配置 (仅当 CLI 参数不为 None 时)
    for key, value in vars(args).items():
        if key != 'config' and value is not None:
            final_config[key] = value

    return final_config


def validate_config(config: Dict[str, Any]) -> None:
    """验证必要的配置项是否存在。"""
    required_keys = ['example_path', 'prompt_path', 'output_path', 'model_path']
    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        print(f"Error: Missing required configuration keys: {missing_keys}")
        print("Please provide them via --config YAML file or CLI arguments.")
        sys.exit(1)

if __name__ == '__main__':
    # EXAMPLE_PATH = 'data/MATH/validation.jsonl'
    # PROMPT_PATH = 'cs336_alignment/prompts/r1_zero.prompt'
    # OUTPUT_PATH = 'results/sft_v1_result.jsonl'
    # MODEL_PATH = 'checkpoints/sft_v1'
    config = parse_arguments()
    validate_config(config)

    print("正在用以下配置运行评估")
    for k, v in config.items():
        print(f"{k}: {v}")

    output_dir = os.path.dirname(config['output_path'])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    run_evaluate(
        example_path=config['example_path'],
        prompt_path=config['prompt_path'],
        output_path=config['output_path'],
        model_path=config['model_path'],
        max_tokens=config.get('max_tokens', 1024),
        temperature=config.get('temperature', 1.0)
    )





