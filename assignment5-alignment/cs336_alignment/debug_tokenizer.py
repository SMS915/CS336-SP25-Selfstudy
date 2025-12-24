import os
from transformers.models.auto.tokenization_auto import AutoTokenizer

# ================= 配置区 =================
# 替换为你的模型路径 (Base 模型 或 已保存了 Tokenizer 的 Checkpoint)
# MODEL_PATH = "Qwen/Qwen2.5-Math-1.5B" 
MODEL_PATH = "checkpoints/sft_v4/epoch0" # 示例：指向你的训练输出目录

# 你想要测试的特殊 Token
SPECIAL_TOKENS = ["<think>", "</think>", "<answer>", "</answer>"]

# 测试用例：模拟真实的训练样本
TEST_CASE = (
    "User: Calculate 1+1.\n"
    "Assistant: <think>\n"
    "1 + 1 is 2.\n"
    "</think>\n"
    "<answer>\n"
    "\\boxed{2}\n"
    "</answer>"
)
# =========================================

def print_separator(title):
    print(f"\n{'='*20} {title} {'='*20}")

def debug_tokenizer():
    print(f"正在加载 Tokenizer: {MODEL_PATH} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"加载失败: {e}")
        return

    # 1. 检查 Special Token 是否存在
    print_separator("1. Special Token ID 检查")
    
    # 如果是从 Base 加载，这里模拟添加过程（用于测试效果）
    # 如果是从 Checkpoint 加载，这里应该直接能查到
    tokens_to_add = []
    for token in SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            print(f"⚠️  Token '{token}' 当前不存在，准备添加...")
            tokens_to_add.append(token)
        else:
            print(f"✅ Token '{token}' 已存在，ID: {token_id}")

    if tokens_to_add:
        num_added = tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})
        print(f"模拟添加了 {num_added} 个新 Token。")
        # 重新打印 ID
        for token in tokens_to_add:
            print(f"🆕 Token '{token}' 新分配 ID: {tokenizer.convert_tokens_to_ids(token)}")

    # 2. 编码测试 (Encoding Test)
    print_separator("2. 编码测试 (Encoding Check)")
    print(f"原始文本:\n{repr(TEST_CASE)}")
    
    encoded = tokenizer(TEST_CASE, add_special_tokens=False)
    input_ids = encoded['input_ids']
    
    print(f"\n生成 ID 序列长度: {len(input_ids)}")
    print(f"ID 序列预览: {input_ids}")

    # 关键：检查标签是否被切分
    print("\n--- 关键标签解析详情 ---")
    for tag in SPECIAL_TOKENS:
        tag_id = tokenizer.convert_tokens_to_ids(tag)
        if tag_id in input_ids:
            count = input_ids.count(tag_id)
            print(f"✅ 成功: '{tag}' (ID {tag_id}) 在序列中出现了 {count} 次。")
        else:
            print(f"❌ 失败: '{tag}' 没有作为独立 ID 出现！它被切碎了！")
            # 尝试查找被切碎的片段
            # 比如 <think> 可能会被切成 <, th, ink, >
            print(f"   (这意味着模型训练时没学到这个特殊token)")

    # 3. 解码测试 (Decoding Test)
    print_separator("3. 解码测试 (Decoding Check)")
    
    # 测试 skip_special_tokens 对结果的影响
    decode_with_special = tokenizer.decode(input_ids, skip_special_tokens=False)
    decode_no_special = tokenizer.decode(input_ids, skip_special_tokens=True)
    
    print(f"[skip_special_tokens=False]:\n{repr(decode_with_special)}")
    print("-" * 30)
    print(f"[skip_special_tokens=True]:\n{repr(decode_no_special)}")
    
    if "<think>" not in decode_no_special:
        print("\n💡 提示: 当 skip_special_tokens=True 时，标签消失是正常的。")
        print("   但在 Reward Function 中，务必使用 skip_special_tokens=False！")

    # 4. 模拟 Masking 边界检查 (Simulate Data Collator)
    print_separator("4. Masking 边界模拟")
    
    # 假设 Prompt 是直到 <think> 之前的部分
    prompt_part = "User: Calculate 1+1.\nAssistant: "
    response_start = "<think>"
    
    prompt_ids = tokenizer.encode(prompt_part, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt_part + response_start, add_special_tokens=False)
    
    print(f"Prompt 文本: {repr(prompt_part)}")
    print(f"Prompt IDs 长度: {len(prompt_ids)}")
    print(f"Full (Prompt+<think>) IDs 长度: {len(full_ids)}")
    
    # 检查边界
    new_tokens_count = len(full_ids) - len(prompt_ids)
    new_token_ids = full_ids[len(prompt_ids):]
    
    print(f"\n新增的 Token IDs: {new_token_ids}")
    print(f"新增部分解码: {tokenizer.decode(new_token_ids)}")
    
    think_id = tokenizer.convert_tokens_to_ids("<think>")
    
    if len(new_token_ids) == 1 and new_token_ids[0] == think_id:
        print("✅ 完美: Prompt 和 Response 的边界清晰，<think> 是 Response 的第 1 个 Token。")
        print("   (Masking 时使用 len(prompt_ids) 作为分界线是安全的)")
    else:
        print("⚠️  警告: 边界模糊！")
        if think_id in new_token_ids:
             print("   虽然 <think> 在里面，但可能前面粘连了空格或换行符。")
        else:
             print("   ❌ <think> 被切分了，或者被 Prompt 吞掉了！")

if __name__ == "__main__":
    debug_tokenizer()