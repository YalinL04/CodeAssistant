"""Entry point: run model evaluation on HumanEval."""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on HumanEval")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the trained model",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/eval",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Model Evaluation on HumanEval / HumanEval+")
    print("=" * 60)

    results = evaluate_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
