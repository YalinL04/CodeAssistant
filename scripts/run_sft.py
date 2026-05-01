"""Entry point: run SFT training."""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.sft_trainer import run_sft_training


def main():
    parser = argparse.ArgumentParser(description="Run SFT training")
    parser.add_argument(
        "--config",
        default="configs/sft_config.yaml",
        help="Path to SFT config file",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SFT Training - DeepSeek-Coder-6.7B")
    print("=" * 60)

    model_path = run_sft_training(args.config)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
