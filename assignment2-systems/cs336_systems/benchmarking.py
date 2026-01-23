import yaml
import timeit
import torch
import random
import argparse
import numpy as np

import torch.cuda.nvtx as nvtx
import cs336_basics.model
from cs336_basics.utils import Softmax
from einops import einsum, rearrange
from jaxtyping import Float, Bool
from typing import Optional
from torch.optim import AdamW
from math import sqrt

from cs336_basics.model import ScaledDotProductAttention as original_SDPA


def annotated_scaled_dot_product_attention(
        Q: Float[torch.Tensor, "batch_size num_q q_seq_len d_q"],
        K: Float[torch.Tensor, "batch_size num_k k_seq_len d_k"],
        V: Float[torch.Tensor, "batch_size num_v v_seq_len d_v"],
        mask: Optional[Bool[torch.Tensor, "batch_size q_seq_len k_seq_len"]] = None
) -> Float[torch.Tensor, "batch_size q_seq_len d_v"]:
    with nvtx.range("ScaledDotProductAttention"):
        is_gqa = False
        if Q.ndim == 4:
            num_heads_q, num_heads_k = Q.shape[1], K.shape[1]
            if num_heads_q != num_heads_k:
                is_gqa = True
                n_rep = num_heads_q // num_heads_k
                Q = rearrange(Q, 'b (h_kv n_rep) l d -> b h_kv n_rep l d', n_rep=n_rep)
                K = rearrange(K, 'b h_kv l d -> b h_kv 1 l d', n_rep=n_rep)
                V = rearrange(V, 'b h_kv l d -> b h_kv 1 l d', n_rep=n_rep)

        d_k = K.shape[-1]
        with nvtx.range("Compute Attention Score"):
            attn_scores = einsum(Q, K, "b ... q d, b ... k d -> b ... q k") / sqrt(d_k)

        with nvtx.range("Masking"):
            if mask is not None:
                attn_scores = torch.where(mask, attn_scores, float('-inf'))
        with nvtx.range("Softmax"):
            attn_weights = Softmax(attn_scores, -1)

        with nvtx.range("Final matmul"):
            output = einsum(attn_weights, V, "b ... q k, b ... k d -> b ... q d")
        if is_gqa:
            output = rearrange(output, "b h_kv n_rep l d -> b (h_kv n_rep) l d")

        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='configs/model_config.yaml')

    parser.add_argument("--model_size", type=str, required=True)
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--num_heads", type=int)

    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, required=True)

    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measure_steps", type=int, default=10)

    parser.add_argument("--optimize", action='store_true')
    parser.add_argument('--precision', type=str, default='autocast', choices=['fp32', 'fp16', 'bf16', 'autocast'],
                        help='Training precision. "autocast" will automatically choose the best available.')
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    model_size = args.model_size if args.model_size in config else "small"
    model_config = config[model_size]

    model_args = ['vocab_size', 'd_model', 'd_ff', 'num_heads', 'num_layers', 'context_length']

    for key, value in vars(args).items():
        if key in model_args and value is not None:
            print(f"从命令行覆盖参数{key}")
            model_config[key] = value

    vocab_size = model_config.get('vocab_size', 50257)
    d_model = model_config.get('d_model', 768)
    d_ff = model_config.get('d_ff', 64 * ((round(d_model * 8 / 3) + 63) // 64))
    num_layers = model_config.get('num_layers', 12)
    num_heads = model_config.get('num_heads', d_model // 64)
    context_length = model_config.get('context_length', 512)

    batch_size = args.batch_size
    warmup_steps = args.warmup_steps


    cs336_basics.model.ScaledDotProductAttention = annotated_scaled_dot_product_attention

    cs336_basics.model.ScaledDotProductAttention = annotated_scaled_dot_product_attention

    model = cs336_basics.model.cs336_basics.model.TransformerLMLM(vocab_size=vocab_size,
                                             context_length=context_length,
                                             d_model=d_model,
                                             d_ff=d_ff,
                                             num_layers=num_layers,
                                             num_heads=num_heads)

    model.to('cuda')
    # model.count_params()
    model.to('cuda')
    # model.count_params()
    optimize = args.optimize == True
    if optimize:
        optimizer = AdamW(model.parameters(),
                          lr=0.001,
                          weight_decay=0.1,
                          betas=(0.95, 0.99)
                          )

    amp_enabled = True
    if args.precision == 'fp32':
        amp_enabled = False
        amp_dtype = torch.float32
        use_scaler = False
    elif args.precision == 'fp16':
        amp_dtype = torch.float16
        use_scaler = True
    elif args.precision == 'bf16':
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            use_scaler = False
        else:
            print("Warning: BF16 not supported on this device, falling back to FP16.")
            amp_dtype = torch.float16
            use_scaler = True
    elif args.precision == 'autocast':
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        use_scaler = (amp_dtype == torch.float16)

    if amp_enabled:
        print(f"AMP enabled, dtype: {amp_dtype}, using GradScaler: {use_scaler}")
    else:
        print("AMP disabled, running in FP32.")

    scaler = torch.amp.grad_scaler.GradScaler(enabled=use_scaler)

    batched_data = torch.randint(0, vocab_size, (batch_size, context_length)).to('cuda').to('cuda')

    print("Warmup Stage")
    warmup_start = timeit.default_timer()
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        output = model(batched_data)
        loss = output.sum()
        loss.backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

    # 同步，确保预热完成
    torch.cuda.synchronize()
    warmup_end = timeit.default_timer()
    warmup_time = warmup_end - warmup_start
    print(f"Warmup finish! Takes {warmup_time:.2f} seconds")

    # 正式测量
    measure_steps = args.measure_steps
    forward_timings = []
    backward_timings = []
    optimize_timings = []
    total_timings = []

    print(f"Measuring for {measure_steps} steps")
    for _ in range(measure_steps):
        if optimize:
            optimizer.zero_grad()
        start_time = timeit.default_timer()
        with nvtx.range("Forward Pass"):
            with torch.amp.autocast_mode.autocast(device_type='cuda', enabled=amp_enabled, dtype=amp_dtype):
                output = model(batched_data)
        torch.cuda.synchronize()

        forward_end = timeit.default_timer()
        with nvtx.range("Backward Pass"):
            loss = output.sum()
            scaler.scale(loss).backward()
        torch.cuda.synchronize()
        backward_end = timeit.default_timer()

        if optimize:
            with nvtx.range("Optimizer step"):
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
                optimize_end = timeit.default_timer()

        end_time = timeit.default_timer()

        forward_timings.append(forward_end - start_time)
        backward_timings.append(backward_end - forward_end)
        total_timings.append(end_time - start_time)

        if optimize:
            optimize_timings.append(optimize_end - backward_end)

    forward_mean = np.mean(forward_timings)
    forward_std = np.std(forward_timings)

    backward_mean = np.mean(backward_timings)
    backward_std = np.std(backward_timings)

    total_mean = forward_mean + backward_mean
    total_std = np.std(total_timings)

    if optimize:
        optimize_mean = np.mean(optimize_timings)
        optimize_std = np.mean(optimize_timings)
    print("Measurement ends!")
    print(f"Takes {np.sum(total_timings):.2f} secs in total")
    print(f"For whole progress, the average time is {total_mean:.2f}, standard deviation is {total_std:.2f}.")
    print(f"For forward pass, the average time is {forward_mean:.2f}, standard deviation is {forward_std:.2f}.")
    print(f"For backward pass, the average time is {backward_mean:.2f}, standard deviation is {backward_std:.2f}.")
    if optimize:
        print(f"For optimizer step ,the average time is {optimize_mean:.2f}, standard deviation is {optimize_std:.2f}.")


if __name__ == '__main__':
    main()
