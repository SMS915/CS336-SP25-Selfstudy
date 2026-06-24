import argparse
import json
import os
import re


def extract_tag_content(response: str, start_tag: str, end_tag: str) -> str | None:
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def build_response(think_text: str, answer_text: str, keep_think_tags: bool, keep_answer_tags: bool) -> str:
    think_text = normalize_whitespace(think_text)
    answer_text = normalize_whitespace(answer_text)

    if keep_think_tags:
        reasoning_part = f"<think>\n{think_text}\n</think>"
    else:
        reasoning_part = think_text

    if keep_answer_tags:
        answer_part = f"<answer>{answer_text}</answer>"
    else:
        answer_part = answer_text

    if reasoning_part:
        return f"{reasoning_part}\n\n{answer_part}".strip()
    return answer_part


def convert_dataset(args) -> None:
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total = 0
    kept = 0
    rejected = {
        "json_error": 0,
        "missing_prompt": 0,
        "missing_tags": 0,
        "missing_boxed": 0,
        "empty_content": 0,
    }

    with open(args.input_path, "r", encoding="utf-8") as fin, open(args.output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                rejected["json_error"] += 1
                continue

            prompt = item.get("prompt") or item.get("question") or item.get("problem")
            response = item.get("response") or item.get("solution")
            if not prompt or not response:
                rejected["missing_prompt"] += 1
                continue

            think_text = extract_tag_content(response, "<think>", "</think>")
            answer_text = extract_tag_content(response, "<answer>", "</answer>")
            if think_text is None or answer_text is None:
                rejected["missing_tags"] += 1
                continue

            if "\\boxed" not in answer_text:
                rejected["missing_boxed"] += 1
                continue

            final_response = build_response(
                think_text=think_text,
                answer_text=answer_text,
                keep_think_tags=args.keep_think_tags,
                keep_answer_tags=args.keep_answer_tags,
            )
            if not final_response.strip():
                rejected["empty_content"] += 1
                continue

            out_item = {"prompt": prompt, "response": final_response}
            fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            kept += 1
            if args.max_samples > 0 and kept >= args.max_samples:
                break

    print("=== Qwen3 SFT Data Build ===")
    print(f"input_path: {args.input_path}")
    print(f"output_path: {args.output_path}")
    print(f"kept: {kept}")
    print(f"total_seen: {total}")
    print(f"rejected: {rejected}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a Qwen3-friendly SFT dataset from formatted CoT data.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/CoT/Bespoke-Stratos-17k-formatted.jsonl",
        help="Input JSONL with complete <think>/<answer> tags.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/CoT/Bespoke-Stratos-17k-qwen3-sft.jsonl",
        help="Output JSONL for SFT.",
    )
    parser.add_argument("--keep_think_tags", action="store_true", help="Keep <think> tags in the output response.")
    parser.add_argument("--keep_answer_tags", action="store_true", help="Keep <answer> tags in the output response.")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional output sample cap.")
    return parser.parse_args()


if __name__ == "__main__":
    convert_dataset(parse_args())
