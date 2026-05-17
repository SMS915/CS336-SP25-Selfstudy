from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import torch
import triton.testing

from cs336_systems.FlashAttention import flash_attention_torch, flash_fwd_kernel


DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


@dataclass(frozen=True)
class AttentionImpl:
    name: str
    forward_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor]
    supports_backward: bool


def parse_csv_ints(text: str) -> list[int]:
    return [int(piece.strip()) for piece in text.split(",") if piece.strip()]


def parse_csv_strs(text: str) -> list[str]:
    return [piece.strip() for piece in text.split(",") if piece.strip()]


def regular_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = q @ k.transpose(-2, -1) * scale
    if is_causal:
        q_idx = torch.arange(q.shape[-2], device=q.device)
        k_idx = torch.arange(k.shape[-2], device=k.device)
        causal_mask = q_idx[:, None] >= k_idx[None, :]
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), -1e6)
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def flash_torch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
) -> torch.Tensor:
    return flash_attention_torch.apply(q, k, v, is_causal)


def flash_triton_forward_only(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    q_tile_size: int = 32,
    k_tile_size: int = 32,
) -> torch.Tensor:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("flash_triton_forward_only expects 3D tensors of shape (batch, seq, d).")

    batch_size, n_queries, d = q.shape
    n_keys = k.shape[1]
    # if n_queries % q_tile_size != 0 or n_keys % k_tile_size != 0:
    #     raise ValueError(
    #         f"Current Triton kernel assumes seq lengths divisible by tile size. "
    #         f"Got n_queries={n_queries}, n_keys={n_keys}, "
    #         f"q_tile_size={q_tile_size}, k_tile_size={k_tile_size}."
    #     )

    o = torch.empty_like(q)
    l = torch.empty(batch_size, n_queries, device=q.device, dtype=torch.float32)
    grid = (n_queries // q_tile_size, batch_size)
    scale = 1.0 / math.sqrt(d)

    flash_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        l,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        l.stride(0),
        l.stride(1),
        n_queries,
        n_keys,
        scale,
        d,
        q_tile_size,
        k_tile_size,
        is_causal,
    )
    return o


IMPLEMENTATIONS: dict[str, AttentionImpl] = {
    "regular_torch": AttentionImpl("regular_torch", regular_attention, True),
    "flash_torch": AttentionImpl("flash_torch", flash_torch_attention, True),
    "flash_triton": AttentionImpl("flash_triton", flash_triton_forward_only, False),
}


def benchmark_forward(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    warmup: int,
    rep: int,
) -> float:
    def run() -> None:
        with torch.no_grad():
            fn(q, k, v, is_causal)

    return triton.testing.do_bench(run, warmup=warmup, rep=rep, return_mode="mean")


def benchmark_backward(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    warmup: int,
    rep: int,
) -> float:
    output = fn(q, k, v, is_causal)
    do = torch.randn_like(output)

    def run() -> None:
        output.backward(do, retain_graph=True)

    return triton.testing.do_bench(
        run,
        warmup=warmup,
        rep=rep,
        grad_to_none=[q, k, v],
        return_mode="mean",
    )


def benchmark_end_to_end(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    warmup: int,
    rep: int,
) -> float:
    do = torch.randn_like(q)

    def run() -> None:
        output = fn(q, k, v, is_causal)
        output.backward(do)

    return triton.testing.do_bench(
        run,
        warmup=warmup,
        rep=rep,
        grad_to_none=[q, k, v],
        return_mode="mean",
    )


def make_qkv(
    batch_size: int,
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=requires_grad)
    return q, k, v


def benchmark_one_impl(
    impl: AttentionImpl,
    batch_size: int,
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    is_causal: bool,
    device: torch.device,
    warmup: int,
    rep: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "impl": impl.name,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "dtype": str(dtype).replace("torch.", ""),
        "causal": is_causal,
        "forward_ms": float("nan"),
        "backward_ms": float("nan"),
        "end_to_end_ms": float("nan"),
        "status": "ok",
    }

    try:
        q, k, v = make_qkv(batch_size, seq_len, d_model, dtype, device, requires_grad=False)
        row["forward_ms"] = benchmark_forward(impl.forward_fn, q, k, v, is_causal, warmup, rep)
    except Exception as exc:  # noqa: BLE001
        row["status"] = f"forward_failed: {type(exc).__name__}"
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return row

    if not impl.supports_backward:
        row["status"] = "forward_only"
        return row

    try:
        q, k, v = make_qkv(batch_size, seq_len, d_model, dtype, device, requires_grad=True)
        row["backward_ms"] = benchmark_backward(impl.forward_fn, q, k, v, is_causal, warmup, rep)

        q, k, v = make_qkv(batch_size, seq_len, d_model, dtype, device, requires_grad=True)
        row["end_to_end_ms"] = benchmark_end_to_end(impl.forward_fn, q, k, v, is_causal, warmup, rep)
    except Exception as exc:  # noqa: BLE001
        row["status"] = f"backward_failed: {type(exc).__name__}"
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return row


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark regular attention and FlashAttention implementations.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_lengths", type=str, default="128,256,512,1024,2048,4096,8192")
    parser.add_argument("--d_models", type=str, default="16,32,64,128")
    parser.add_argument("--dtypes", type=str, default="fp32,bf16")
    parser.add_argument("--impls", type=str, default="regular_torch,flash_triton")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--csv_path", type=str, default="")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This benchmarking script is intended for CUDA devices.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    seq_lengths = parse_csv_ints(args.seq_lengths)
    d_models = parse_csv_ints(args.d_models)
    dtype_names = parse_csv_strs(args.dtypes)
    impl_names = parse_csv_strs(args.impls)

    unknown_impls = [name for name in impl_names if name not in IMPLEMENTATIONS]
    if unknown_impls:
        raise ValueError(f"Unknown implementations: {unknown_impls}")

    dtype_values = []
    for dtype_name in dtype_names:
        if dtype_name not in DTYPE_MAP:
            raise ValueError(f"Unsupported dtype name: {dtype_name}")
        dtype_values.append(DTYPE_MAP[dtype_name])

    torch.manual_seed(args.seed)
    torch.cuda.set_device(device)

    rows: list[dict[str, object]] = []
    for dtype in dtype_values:
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            rows.append(
                {
                    "impl": "all",
                    "batch_size": args.batch_size,
                    "seq_len": -1,
                    "d_model": -1,
                    "dtype": "bfloat16",
                    "causal": args.causal,
                    "forward_ms": float("nan"),
                    "backward_ms": float("nan"),
                    "end_to_end_ms": float("nan"),
                    "status": "skipped_bf16_unsupported",
                }
            )
            continue

        for seq_len in seq_lengths:
            for d_model in d_models:
                for impl_name in impl_names:
                    impl = IMPLEMENTATIONS[impl_name]
                    print(
                        f"Benchmarking impl={impl.name} seq_len={seq_len} "
                        f"d_model={d_model} dtype={str(dtype).replace('torch.', '')}"
                    )
                    row = benchmark_one_impl(
                        impl=impl,
                        batch_size=args.batch_size,
                        seq_len=seq_len,
                        d_model=d_model,
                        dtype=dtype,
                        is_causal=args.causal,
                        device=device,
                        warmup=args.warmup,
                        rep=args.rep,
                    )
                    rows.append(row)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if args.csv_path:
        df.to_csv(args.csv_path, index=False)
        print(f"Saved benchmark results to {args.csv_path}")


if __name__ == "__main__":
    main()
