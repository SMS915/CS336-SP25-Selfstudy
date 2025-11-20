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
    print(f"开始生成{len(prompts)}条数据")
    start_time = time.time()
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    end_time = time.time()
    print(f"生成完成，共用时{end_time - start_time}秒")
    results = []
    correct_count = 0
    ans_error_count = 0
    format_error_count = 0
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

def run_evalute(example_path: str, prompt_path: str, output_path: str, model_path: str):
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
    OUTPUT_PATH = 'results/baseline_result.jsonl'
    MODEL_PATH = 'models/Qwen2.5-Math-1.5B'
    os.makedirs('results', exist_ok=True)
    run_evalute(EXAMPLE_PATH, PROMPT_PATH, OUTPUT_PATH, MODEL_PATH)





