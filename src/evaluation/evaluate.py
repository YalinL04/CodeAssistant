"""Run evalplus evaluation on generated samples."""

import subprocess
import sys
import re
from pathlib import Path


def run_evalplus(
    samples_file: str,
    dataset: str = "humaneval",
) -> dict:
    """
    Run evalplus evaluation on generated samples.

    Args:
        samples_file: Path to JSONL file with generated samples
        dataset: 'humaneval' or 'mbpp'

    Returns:
        Dict with evaluation results {'pass@1': float, ...}
    """
    if not Path(samples_file).exists():
        raise FileNotFoundError(f"Samples file not found: {samples_file}")

    cmd = [
        sys.executable, "-m", "evalplus.evaluate",
        "--dataset", dataset,
        "--samples", samples_file,
    ]

    print(f"Running evalplus: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    print(result.stdout)
    if result.stderr:
        print(f"stderr: {result.stderr}")

    # Parse results from output
    results = parse_evalplus_output(result.stdout)
    return results


def parse_evalplus_output(output: str) -> dict:
    """
    Parse evalplus output to extract pass@k scores.

    Expected format:
    humaneval (base tests)                                                                                                                                                                                              
    pass@1: 0.xxx                                                                                                                                                                                                       
    humaneval+ (base + extra tests)                                                                                                                                                                                     
    pass@1: 0.xxx
    """
    results = {}
    # Match base results (HumanEval)
    base_match = re.search(
        r"humaneval \(base tests\)\s*pass@1:\s*([0-9.]+)",
        output
    )
    if base_match:
        results["humaneval"] = float(base_match.group(1))

    # Match base + extra results (HumanEval+)
    extra_match = re.search(
        r"humaneval\+ \(base \+ extra tests\)\s*pass@1:\s*([0-9.]+)",
        output
    )
    if extra_match:
        results["humaneval+"] = float(extra_match.group(1))

    return results


def evaluate_model(
    model_path: str,
    output_dir: str = "outputs/eval",
    max_new_tokens: int = 1024,
) -> dict:
    """
    Full evaluation pipeline: generate samples then run evalplus.

    Args:
        model_path: Path to the trained model
        output_dir: Directory for evaluation outputs
        max_new_tokens: Max tokens for generation

    Returns:
        Dict with evaluation results
    """
    from src.evaluation.generate_samples import generate_samples

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    samples_file = str(output_path / "samples.jsonl")

    # Generate samples
    print("=" * 60)
    print("Step 1: Generating samples")
    print("=" * 60)
    generate_samples(
        model_path=model_path,
        output_file=samples_file,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
    )

    # Run evalplus on HumanEval
    print("\n" + "=" * 60)
    print("Step 2: Evaluating on HumanEval")
    print("=" * 60)
    results = run_evalplus(samples_file, dataset="humaneval")
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    if "humaneval" in results:
        print(f"HumanEval : {results['humaneval']}")
    if "humaneval+" in results:
        print(f"HumanEval+: {results['humaneval+']}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model on HumanEval")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="outputs/eval")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
    )
