import os
import json
import time
from typing import List, Dict, Callable
from vllm import LLM, SamplingParams

# 引入评分函数
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
        # 替换占位符，确保和训练时的格式一致
        prompt = prompt_template.replace("{question}", ex["problem"])
        prompts.append(prompt)
    return prompts

def evaluate_sft_model():
    # ================= 配置区域 =================
    # 1. 指向你训练保存的 Checkpoint 目录
    MODEL_PATH = "checkpoints/sft_v1" 
    
    # 2. 数据路径
    DATA_PATH = "data/MATH/validation.jsonl"
    PROMPT_PATH = "cs336_alignment/prompts/r1_zero.prompt"
    OUTPUT_FILE = "results/sft_eval_results.jsonl"
    
    # 3. 关键：给够生成长度！
    MAX_TOKENS = 4096 
    # ===========================================

    print(f"🚀 Loading SFT model from: {MODEL_PATH}")
    
    # 1. 初始化 vLLM
    # trust_remote_code=True 即使加载本地模型有时也需要，取决于 config
    # 显存利用率设为 0.9，因为现在只跑推理，不训练
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        max_model_len=MAX_TOKENS  # 确保 KV Cache 预留够空间
    )

    # 2. 准备数据
    print("Loading data and prompts...")
    examples = load_data(DATA_PATH)
    
    # 建议先跑前 50-100 条看看效果，全量 5000 条可能要跑一会儿
    # examples = examples[:100] 
    
    with open(PROMPT_PATH, "r") as f:
        prompt_template = f.read()
    
    prompts = formatting_prompt(examples, prompt_template)

    # 3. 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.6, # SFT 后通常可以稍微降低一点温度，或者保持 1.0
        top_p=0.95,
        max_tokens=MAX_TOKENS,
        stop=["</answer>"],             # 遇到此标记停止
        include_stop_str_in_output=True # 保留标记
    )

    # 4. 生成
    print(f"开始生成{len(prompts)} 条数据")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    print(f"生成完成，共用时{end_time - start_time:.2f}秒")

    # 5. 评分
    results = []
    correct_count = 0
    format_error_count = 0
    ans_error_count = 0
    
    print("📊 Scoring...")
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        example = examples[i]
        truth = example["solution"]
        text_for_grading = generated_text.replace("</think><answer>", "</think> <answer>")
        # 评分
        # 注意：generated_text 开头可能没有 <think> (因为它在 prompt 里)
        # 但 grader 主要看 <answer>，通常没问题。
        metrics = r1_zero_reward_fn(text_for_grading, truth)
        if metrics.get("reward", 0.0) == 1.0:
            correct_count += 1
        elif metrics.get("format_reward", 0.0) == 1.0:
            ans_error_count += 1
        else:
            format_error_count += 1

        results.append({
            "problem": example["problem"],
            "gold_solution": truth,
            "generated_text": generated_text,
            "metrics": metrics
        })
    # 6. 打印报告
    accuracy = correct_count / len(prompts)

    
    print("\n" + "="*30)
    print("评估结果如下:")
    print(f"Model: {MODEL_PATH}")
    print(f"完全正确: {accuracy: .2%}")
    print(f"格式正确，答案错误: {ans_error_count / len(prompts):.2%}")
    print(f"格式错误: {format_error_count / len(prompts):.2%}")
    print("="*30 + "\n")

    # 7. 保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    print(f" 结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    evaluate_sft_model()