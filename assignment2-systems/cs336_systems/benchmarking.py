import yaml
import timeit
import argparse
from cs336_basics.model import TransformerLM as Transformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='configs/model_config.yaml')

    parser.add_argument("--model_size", type=str, required=True)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--num_heads", type=int)

    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, required=True)

    
