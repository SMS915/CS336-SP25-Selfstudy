import argparse

from cs336_alignment.bootstrap_runtime import bootstrap_cuda_visible_devices

bootstrap_cuda_visible_devices(default_config_path="configs/train/grpo_multigpu_config.yaml")

from cs336_alignment.train_grpo import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/grpo_multigpu_config.yaml")
    args = parser.parse_args()
    train(args.config)
