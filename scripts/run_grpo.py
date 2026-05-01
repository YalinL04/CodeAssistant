"""Entry point: run GRPO training."""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.grpo_data_builder import build_grpo_dataset
from src.training.grpo_trainer import run_grpo_training


def main():
    parser = argparse.ArgumentParser(description="Run GRPO training")
    parser.add_argument(
        "--config",
        default="configs/grpo_config.yaml",
        help="Path to GRPO config file",
    )
    parser.add_argument(
        "--build_dataset",
        action="store_true",
        help="Build GRPO dataset before training",
    )
    parser.add_argument(
        "--dataset",
        default="mbpp",
        choices=["mbpp", "humaneval"],
        help="Dataset source used when rebuilding the GRPO prompt file",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split used when rebuilding the GRPO prompt file",
    )
    args = parser.parse_args()

    if args.build_dataset:
        print("=" * 60)
        print(f"Building GRPO Dataset from {args.dataset}:{args.split}")
        print("=" * 60)
        build_grpo_dataset(dataset_name=args.dataset, split=args.split)
        print()

    print("=" * 60)
    print("GRPO Training - DeepSeek-Coder-6.7B")
    print("=" * 60)

    model_path = run_grpo_training(args.config)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
