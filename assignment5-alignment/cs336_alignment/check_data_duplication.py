import json
import argparse
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
import re

# --- 配置 ---
# 这里的路径需要替换成你实际的文件路径
DATASETS = {
    "MATH_500": "data/MATH/math500-test.jsonl",
    "MATH_FULL_TEST": "data/MATH/validation.jsonl", # 或者 test.jsonl
    "SFT_TRAIN": "data/MATH/sft_v5_platinum.jsonl"     # 你的 SFT 训练数据
}

def normalize_text(text):
    """
    文本标准化：去除空格、标点、大小写，只保留核心字符。
    这样可以防止因为多一个空格导致匹配失败。
    """
    if not text: return ""
    # 转小写
    text = text.lower()
    # 去除LaTeX命令中的空格等干扰，保留字母数字
    text = re.sub(r'\s+', '', text)
    return text

def get_minhash(text, num_perm=128):
    """生成文本的 MinHash 签名"""
    m = MinHash(num_perm=num_perm)
    # 使用 n-gram (这里用 3-gram)
    width = 3
    text = normalize_text(text)
    if len(text) < width:
        m.update(text.encode('utf8'))
    else:
        for i in range(len(text) - width + 1):
            m.update(text[i:i+width].encode('utf8'))
    return m

def load_dataset_signatures(file_path, name):
    print(f"正在加载并计算指纹: {name} ({file_path})...")
    signatures = []
    problems = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                # 兼容不同的字段名
                problem = data.get('problem') or data.get('prompt') or data.get('question')
                if not problem: continue
                
                # 如果是 SFT 数据，prompt 可能包含 "User: ... \nAssistant:"，需要提取核心问题
                if "User:" in problem and "Assistant:" in problem:
                    # 粗略提取 User 和 Assistant 中间的部分
                    try:
                        problem = problem.split("User:", 1)[1].split("Assistant:", 1)[0].strip()
                    except:
                        pass # 提取失败就用原文本
                
                m = get_minhash(problem)
                signatures.append(m)
                problems.append(problem)
    except FileNotFoundError:
        print(f"⚠️ 文件未找到: {file_path}")
        return [], []
        
    print(f"  - 已加载 {len(signatures)} 条数据")
    return signatures, problems

def check_overlap(source_name, source_sigs, target_name, target_sigs, threshold=0.95):
    """
    检查 Source 中的数据有多少出现在 Target 中
    threshold: 相似度阈值，0.95 表示几乎完全一样
    """
    print(f"\n正在检查 {source_name} -> {target_name} 的重合度...")
    
    # 构建 LSH 索引
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for i, m in enumerate(target_sigs):
        lsh.insert(f"tgt_{i}", m)
    
    overlap_count = 0
    
    for i, m in enumerate(tqdm(source_sigs, desc="Matching")):
        result = lsh.query(m)
        if len(result) > 0:
            overlap_count += 1
            
    ratio = overlap_count / len(source_sigs) if len(source_sigs) > 0 else 0
    print(f"结果: {overlap_count} / {len(source_sigs)} ({ratio:.2%}) 的 {source_name} 样本出现在 {target_name} 中。")
    return ratio

def check_self_duplication(name, sigs, threshold=0.98):
    """检查内部重复"""
    print(f"\n正在检查 {name} 的内部重复...")
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    
    duplicates = 0
    seen = set()
    
    for i, m in enumerate(tqdm(sigs, desc="Self-Check")):
        result = lsh.query(m)
        # result 会包含自己，所以如果 len > 1 说明有重复
        # 且不仅要看数量，还要看是否之前见过
        
        # 简单的去重统计逻辑
        if i in seen:
            continue
            
        real_dupes = [x for x in result if x != f"doc_{i}"]
        if len(real_dupes) > 0:
            duplicates += 1
            # 标记已发现的重复项
            # 这里简化处理，只统计有多少个样本是不唯一的
        
        lsh.insert(f"doc_{i}", m)
        
    # 注意：这个统计是近似的，精确去重需要更复杂的逻辑
    # 这里我们只关心是否有大量重复
    print(f"结果: 发现潜在重复样本。") 

def main():
    # 1. 加载所有数据集的 MinHash
    db = {}
    for name, path in DATASETS.items():
        sigs, texts = load_dataset_signatures(path, name)
        db[name] = {"sigs": sigs, "texts": texts}

    # 2. 关键检查：MATH-500 是否是 Full Test 的子集？
    if db["MATH_500"]["sigs"] and db["MATH_FULL_TEST"]["sigs"]:
        check_overlap("MATH_500", db["MATH_500"]["sigs"], 
                      "MATH_FULL_TEST", db["MATH_FULL_TEST"]["sigs"])

    # 3. 致命检查：数据泄露 (Data Leakage)
    # MATH-500 是否出现在了 SFT 训练集中？
    if db["MATH_500"]["sigs"] and db["SFT_TRAIN"]["sigs"]:
        print("\n🚨 [关键检查] 数据泄露检测: 测试集是否混入了训练集？")
        ratio = check_overlap("MATH_500", db["MATH_500"]["sigs"], 
                              "SFT_TRAIN", db["SFT_TRAIN"]["sigs"])
        if ratio > 0.05:
            print("⚠️  警告: 存在显著的数据泄露风险！模型可能背过了测试题。")
        else:
            print("✅ 安全: 未发现明显的数据泄露。")

    # 4. 训练集内部去重检查
    if db["SFT_TRAIN"]["sigs"]:
        # 简单检查是否有完全重复的 Prompt
        prompts = db["SFT_TRAIN"]["texts"]
        unique_prompts = set(prompts)
        dupe_count = len(prompts) - len(unique_prompts)
        print(f"\nSFT 训练集内部精确重复检查:")
        print(f"总数: {len(prompts)}, 唯一数: {len(unique_prompts)}")
        print(f"重复数: {dupe_count} ({dupe_count/len(prompts):.2%})")

if __name__ == "__main__":
    main()