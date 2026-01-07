import json
import os
import random
import uuid

# ================= 配置区域 =================

# 1. 输入文件路径
PATHS = {
    'local_gsm8k': 'data/GSM8K/train_clean.jsonl',
    'local_math': 'data/MATH/train_split.jsonl',
    'numina_L1': 'data/NuminaMath-1.5/split_by_difficulty/numina_level_1_elementary.jsonl',
    'numina_L2': 'data/NuminaMath-1.5/split_by_difficulty/numina_level_2_intermediate.jsonl',
    'numina_L3': 'data/NuminaMath-1.5/split_by_difficulty/numina_level_3_advanced.jsonl',
    'numina_L4': 'data/NuminaMath-1.5/split_by_difficulty/numina_level_4_expert.jsonl',
}

# 2. 输出路径
OUTPUT_FILE = 'data/NuminaMath-1.5/grpo_final_curriculum.jsonl'

# 3. 总数据量目标 (设为 4800，处于 4000-5000 区间，且易于被 8, 16, 32 整除)
TARGET_TOTAL = 4800


# ===========================================

def standardize_item(raw_item, source_tag, difficulty_level):
    """统一数据格式"""
    problem = raw_item.get('problem') or raw_item.get('question') or raw_item.get('query')
    answer = raw_item.get('answer') or raw_item.get('response')

    if not problem or not answer:
        return None

    orig_id = raw_item.get('id') or raw_item.get('problem_id')
    final_id = f"{source_tag}_{orig_id}" if orig_id else f"{source_tag}_{uuid.uuid4().hex[:8]}"

    return {
        "id": final_id,
        "difficulty": difficulty_level,
        "problem": str(problem).strip(),
        "answer": str(answer).strip()
    }


def load_pool(filepath, source_tag, difficulty_level, limit=None):
    pool = []
    if not os.path.exists(filepath):
        print(f"⚠️  跳过: {filepath}")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                raw = json.loads(line)
                item = standardize_item(raw, source_tag, difficulty_level)
                if item: pool.append(item)
            except:
                continue

    if limit and len(pool) > limit:
        pool = random.sample(pool, limit)

    print(f"📥 {source_tag}: 加载 {len(pool)} 条")
    return pool


def build_curriculum():
    # 1. 构建四个难度池
    # Difficulty 1: 基础 (GSM8K + Numina L1)
    pool_a = load_pool(PATHS['local_gsm8k'], 'gsm8k', 1)
    pool_a += load_pool(PATHS['numina_L1'], 'numina_L1', 1, limit=1000)

    # Difficulty 2: 中级 (MATH + Numina L2)
    pool_b = load_pool(PATHS['local_math'], 'math', 2)
    pool_b += load_pool(PATHS['numina_L2'], 'numina_L2', 2, limit=2000)

    # Difficulty 3: 高级 (Numina L3 - AMC 级别)
    pool_c = load_pool(PATHS['numina_L3'], 'numina_L3', 3)

    # Difficulty 4: 专家 (Numina L4 - AIME 级别)
    pool_d = load_pool(PATHS['numina_L4'], 'numina_L4', 4, limit=5000)

    for p in [pool_a, pool_b, pool_c, pool_d]:
        random.shuffle(p)

    # 2. 课程阶段定义
    # 我们将 4800 条数据分为三个阶段，模拟从易到难的训练过程
    stages = [
        {
            "name": "Stage_1_Warmup",
            "count": int(TARGET_TOTAL * 0.25),  # 1200 条
            "ratios": {'A': 0.40, 'B': 0.40, 'C': 0.20, 'D': 0.00}
        },
        {
            "name": "Stage_2_Climbing",
            "count": int(TARGET_TOTAL * 0.45),  # 2160 条
            "ratios": {'A': 0.10, 'B': 0.30, 'C': 0.40, 'D': 0.20}
        },
        {
            "name": "Stage_3_Mastery",
            "count": int(TARGET_TOTAL * 0.30),  # 1440 条
            "ratios": {'A': 0.00, 'B': 0.10, 'C': 0.40, 'D': 0.50}
        }
    ]

    final_dataset = []
    cursors = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    pools_map = {'A': pool_a, 'B': pool_b, 'C': pool_c, 'D': pool_d}

    print("\n🚀 正在按课程梯度合成数据...")

    for stage in stages:
        stage_items = []
        counts = {k: int(stage['count'] * v) for k, v in stage['ratios'].items()}

        # 修正取整误差
        diff = stage['count'] - sum(counts.values())
        if diff > 0: counts['C'] += diff

        for k, req_count in counts.items():
            if req_count == 0: continue
            pool = pools_map[k]

            # 如果池子数据不足，循环采样
            for _ in range(req_count):
                stage_items.append(pool[cursors[k] % len(pool)])
                cursors[k] += 1

        random.shuffle(stage_items)  # 阶段内打乱
        final_dataset.extend(stage_items)
        print(
            f"✅ {stage['name']}: {len(stage_items)} 条 (A:{counts['A']}, B:{counts['B']}, C:{counts['C']}, D:{counts['D']})")

    # 3. 导出
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in final_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("\n" + "=" * 30)
    print(f"总计生成: {len(final_dataset)} 条")
    print(f"输出路径: {OUTPUT_FILE}")
    print("训练建议: 在 DataLoader 中设置 shuffle=False 以维持难度梯度。")
    print("=" * 30)


if __name__ == "__main__":
    random.seed(42)
    build_curriculum()