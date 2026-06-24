import argparse
import json
import os

import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from cs336_alignment.bootstrap_runtime import bootstrap_cuda_visible_devices
from cs336_alignment.device_config import (
    apply_runtime_environment,
    build_model_load_kwargs,
    resolve_torch_device,
)

bootstrap_cuda_visible_devices()


def load_yaml_like_runtime_args(args: argparse.Namespace) -> dict:
    return {
        "runtime": {
            "device": args.device,
            "cuda_visible_devices": args.cuda_visible_devices,
            "hf_device_map": None,
            "max_memory_gb_per_gpu": args.max_memory_gb_per_gpu,
        },
        "model": {
            "dtype": args.dtype,
            "trust_remote_code": True,
            "attn_implementation": args.attn_implementation,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to a LoRA adapter directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged full model.")
    parser.add_argument("--base_model_path", type=str, default=None, help="Optional override for base model path.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Torch dtype, e.g. bfloat16 or float16.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda or cpu.")
    parser.add_argument("--cuda_visible_devices", type=str, default=None, help="Visible GPU ids, e.g. 7 or 6,7.")
    parser.add_argument("--max_memory_gb_per_gpu", type=int, default=None, help="Optional HF max_memory per visible GPU.")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="HF attention backend, e.g. flash_attention_2 or sdpa.",
    )
    args = parser.parse_args()

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("合并 LoRA adapter 需要 `peft`。请先安装对应依赖。") from exc

    adapter_config_path = os.path.join(args.adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"未找到 adapter_config.json: {adapter_config_path}")

    with open(adapter_config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    base_model_path = args.base_model_path or adapter_config.get("base_model_name_or_path")
    if not base_model_path:
        raise ValueError("无法从 adapter_config.json 推断 base_model_path，请手动传 --base_model_path。")

    runtime_config = load_yaml_like_runtime_args(args)
    apply_runtime_environment(runtime_config)
    default_device = resolve_torch_device(runtime_config)

    print(f"Base model: {base_model_path}")
    print(f"Adapter: {args.adapter_path}")
    print(f"Output: {args.output_path}")

    model_load_kwargs = build_model_load_kwargs(runtime_config)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_load_kwargs)
    if "device_map" not in model_load_kwargs:
        model.to(default_device)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print("Merging adapter weights into base model...")
    merged_model = model.merge_and_unload()

    os.makedirs(args.output_path, exist_ok=True)
    merged_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"Merged model saved to: {args.output_path}")


if __name__ == "__main__":
    main()
