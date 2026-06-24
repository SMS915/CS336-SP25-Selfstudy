import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from cs336_alignment.drgrpo_grader import extract_answer


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: str
    default_count: int


SOURCE_SPECS: Dict[str, SourceSpec] = {
    "math": SourceSpec(
        name="math",
        path="data/MATH/train_split.jsonl",
        default_count=7500,
    ),
    "gsm8k": SourceSpec(
        name="gsm8k",
        path="data/gsm8k/train_clean.jsonl",
        default_count=3000,
    ),
    "numina": SourceSpec(
        name="numina",
        path="data/NuminaMath-1.5/numina_cleaned.jsonl",
        default_count=4500,
    ),
    "curriculum": SourceSpec(
        name="curriculum",
        path="data/NuminaMath-1.5/grpo_final_curriculum.jsonl",
        default_count=0,
    ),
}


def normalize_problem_key(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip().lower()
    return collapsed


def extract_problem(item: Dict[str, Any]) -> str | None:
    for key in ["problem", "question", "prompt", "query"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extract_clean_answer(item: Dict[str, Any]) -> str | None:
    answer = item.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    if isinstance(answer, (int, float)):
        return str(answer)

    solution = item.get("solution")
    if isinstance(solution, str) and solution.strip():
        extracted = extract_answer(solution)
        if extracted is not None and str(extracted).strip():
            return str(extracted).strip()

    response = item.get("response")
    if isinstance(response, str) and response.strip():
        extracted = extract_answer(response)
        if extracted is not None and str(extracted).strip():
            return str(extracted).strip()

    return None


def standardize_item(item: Dict[str, Any], source_name: str, row_idx: int) -> Dict[str, Any] | None:
    problem = extract_problem(item)
    answer = extract_clean_answer(item)
    if problem is None or answer is None:
        return None

    standardized = {
        "id": str(item.get("id") or f"{source_name}_{row_idx:07d}"),
        "source": source_name,
        "problem": problem,
        "answer": answer,
    }

    if "level" in item:
        standardized["level"] = item["level"]
    if "type" in item:
        standardized["type"] = item["type"]
    if "difficulty" in item:
        standardized["difficulty"] = item["difficulty"]
    if "source" in item and item["source"] != source_name:
        standardized["raw_source"] = item["source"]
    if "synthetic" in item:
        standardized["synthetic"] = item["synthetic"]

    return standardized


def load_source_pool(spec: SourceSpec) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = {
        "rows_seen": 0,
        "rows_kept": 0,
        "rows_invalid": 0,
        "rows_duplicate_within_source": 0,
    }
    pool: List[Dict[str, Any]] = []
    seen_problem_keys = set()

    if not os.path.exists(spec.path):
        raise FileNotFoundError(f"Source file not found: {spec.path}")

    with open(spec.path, "r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if not line.strip():
                continue
            stats["rows_seen"] += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                stats["rows_invalid"] += 1
                continue

            standardized = standardize_item(item, spec.name, row_idx)
            if standardized is None:
                stats["rows_invalid"] += 1
                continue

            problem_key = normalize_problem_key(standardized["problem"])
            if problem_key in seen_problem_keys:
                stats["rows_duplicate_within_source"] += 1
                continue
            seen_problem_keys.add(problem_key)

            pool.append(standardized)
            stats["rows_kept"] += 1

    return pool, stats


def sample_pool(pool: List[Dict[str, Any]], requested_count: int, rng: random.Random) -> List[Dict[str, Any]]:
    if requested_count <= 0 or requested_count >= len(pool):
        return list(pool)
    return rng.sample(pool, requested_count)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a mixed RL dataset for Qwen3 math alignment.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/RL/qwen3_rl_math_mix_15k.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--math_count", type=int, default=SOURCE_SPECS["math"].default_count)
    parser.add_argument("--gsm8k_count", type=int, default=SOURCE_SPECS["gsm8k"].default_count)
    parser.add_argument("--numina_count", type=int, default=SOURCE_SPECS["numina"].default_count)
    parser.add_argument("--curriculum_count", type=int, default=SOURCE_SPECS["curriculum"].default_count)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle final dataset before writing.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    requested_counts = {
        "math": max(args.math_count, 0),
        "gsm8k": max(args.gsm8k_count, 0),
        "numina": max(args.numina_count, 0),
        "curriculum": max(args.curriculum_count, 0),
    }

    rng = random.Random(args.seed)
    selected_items: List[Dict[str, Any]] = []
    seen_global_problem_keys = set()
    source_stats: Dict[str, Dict[str, int]] = {}

    for source_name in ["math", "gsm8k", "numina", "curriculum"]:
        requested_count = requested_counts[source_name]
        if requested_count == 0:
            continue

        spec = SOURCE_SPECS[source_name]
        pool, load_stats = load_source_pool(spec)
        sampled = sample_pool(pool, requested_count, rng)

        added = 0
        duplicate_across_sources = 0
        for item in sampled:
            problem_key = normalize_problem_key(item["problem"])
            if problem_key in seen_global_problem_keys:
                duplicate_across_sources += 1
                continue
            seen_global_problem_keys.add(problem_key)
            selected_items.append(item)
            added += 1

        source_stats[source_name] = {
            **load_stats,
            "requested_count": requested_count,
            "available_after_cleaning": len(pool),
            "sampled_count": len(sampled),
            "added_to_final": added,
            "duplicate_across_sources": duplicate_across_sources,
        }

    if args.shuffle:
        rng.shuffle(selected_items)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in selected_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_requested = sum(requested_counts.values())
    print("=== Qwen3 RL Data Build ===")
    print(f"output_path: {args.output_path}")
    print(f"seed: {args.seed}")
    print(f"shuffle: {args.shuffle}")
    print(f"requested_total: {total_requested}")
    print(f"final_total: {len(selected_items)}")
    for source_name, stats in source_stats.items():
        print(
            f"[{source_name}] requested={stats['requested_count']}, "
            f"available={stats['available_after_cleaning']}, "
            f"sampled={stats['sampled_count']}, added={stats['added_to_final']}, "
            f"dup_cross_source={stats['duplicate_across_sources']}, invalid={stats['rows_invalid']}"
        )


if __name__ == "__main__":
    main()
