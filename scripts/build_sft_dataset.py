"""Entry point: build mixed SFT dataset from MBPP and CodeAlpaca."""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.sft_data_builder import build_sft_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build SFT training data from MBPP and CodeAlpaca"
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--mbpp_repeat",
        type=int,
        default=5,
        help="How many times to repeat MBPP train samples in the mixed training set",
    )
    parser.add_argument(
        "--codealpaca_val_size",
        type=float,
        default=0.05,
        help="Validation split ratio for CodeAlpaca",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Building SFT Dataset from MBPP + CodeAlpaca")
    print("=" * 60)

    train_file, val_file = build_sft_dataset(
        output_dir=args.output_dir,
        codealpaca_val_size=args.codealpaca_val_size,
        mbpp_repeat=args.mbpp_repeat,
    )

    print("\nDone!")
    print(f"  Train: {train_file}")
    print(f"  Val:   {val_file}")


if __name__ == "__main__":
    main()
