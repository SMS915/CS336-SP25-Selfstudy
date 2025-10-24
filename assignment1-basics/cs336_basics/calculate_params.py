from cs336_basics.TransformerLM import TransformerLM
import torch

def main():
    model = TransformerLM(
    vocab_size=50257,
    context_length=256,
    d_model=512,
    num_layers=10,
    num_heads=8,
    d_ff = None,
    rope_theta=10000.0
    ).to(torch.device('cpu'))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")

if __name__ == "__main__":
    main()