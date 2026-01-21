import yaml
import timeit
import torch
import random
import argparse
import numpy as np

from cs336_basics.model import TransformerLM as Transformer

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

    model = Transformer(vocab_size=vocab_size,                        
                        context_length=context_length,
                        d_model=d_model,
                        d_ff=d_ff,
                        num_layers=num_layers,
                        num_heads=num_heads)
    
    batched_data = torch.randint(0, vocab_size, (batch_size, context_length))

    print("Warmup Stage")
    warmup_start = timeit.default_timer()
    for _ in range(warmup_steps):
        output = model(batched_data)
        loss = output.sum()
        loss.backward()

    # 同步，确保预热完成
    torch.cuda.synchronize()
    warmup_end = timeit.default_timer()
    warmup_time = warmup_end - warmup_start
    print(f"Warmup finish! Takes {warmup_time:.2f} seconds")

    # 正式测量
    measure_steps = args.measure_steps
    forward_timings = []
    backward_timings = []
    total_timings = []
    print(f"Measuring for {measure_steps} steps")
    for _ in range(measure_steps):
        start_time = timeit.default_timer()

        # 运行测量操作
        output = model(batched_data)
        torch.cuda.synchronize()

        forward_end = timeit.default_timer()

        loss = output.sum()
        loss.backward()
        torch.cuda.synchronize()

        end_time = timeit.default_timer()

        forward_timings.append(forward_end - start_time)
        backward_timings.append(end_time - forward_end)
        total_timings.append(end_time - start_time)

    forward_mean = np.mean(forward_timings)
    forward_std = np.std(forward_timings)
    
    backward_mean = np.mean(backward_timings)
    backward_std = np.std(backward_timings)

    total_mean = forward_mean + backward_mean
    total_std = np.std(total_timings)

    print("Measurement ends!")
    print(f"Takes {np.sum(total_timings):.2f} secs in total")
    print(f"For whole progress, the average time is {total_mean:.2f}, standard deviation is {total_std:.2f}.")
    print(f"For forward pass ,the average time is {forward_mean:.2f}, standard deviation is {forward_std:.2f}.")
    print(f"For forward pass ,the average time is {backward_mean:.2f}, standard deviation is {backward_std:.2f}.")


if __name__ == '__main__':
    main()

