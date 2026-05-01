"""Sanity-check GRPO reward discrimination before training."""

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import get_stop_token_ids, load_model_for_inference
from src.rewards.correctness import correctness_reward
from src.rewards.efficiency import efficiency_reward
from src.rewards.readability import readability_reward
from src.training.grpo_trainer import load_grpo_dataset
from src.utils.code_utils import extract_code_from_completion
from src.utils.config_loader import load_config


def _extract_prompt_batch(row_prompt):
    """Normalize a dataset prompt field into message-list format."""
    if isinstance(row_prompt, list):
        return row_prompt
    raise TypeError(f"Unsupported prompt format: {type(row_prompt)!r}")


def _generate_completions(
    model,
    tokenizer,
    prompt_messages: list[dict],
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    """Sample multiple assistant completions for a single GRPO prompt."""
    input_ids = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    stop_token_ids = get_stop_token_ids(tokenizer)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_generations,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids,
        )

    completions = []
    for i in range(num_generations):
        generated_ids = output[i][input_ids.shape[1]:]
        completion_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        completions.append([{"role": "assistant", "content": completion_text}])
    return completions


def _score_prompt_group(
    row: dict,
    completions: list[list[dict]],
    reward_weights: list[float],
    timeout: int,
) -> dict:
    """Compute reward vectors for all sampled completions of one prompt."""
    group_size = len(completions)
    kwargs = {
        "test": [row["test"]] * group_size,
        "entry_point": [row["entry_point"]] * group_size,
        "prompt": [row["prompt"]] * group_size,
        "canonical_solution": [row.get("canonical_solution")] * group_size,
    }

    correctness = correctness_reward(
        completions=completions,
        test=kwargs["test"],
        entry_point=kwargs["entry_point"],
        prompt=kwargs["prompt"],
        timeout=timeout,
    )
    readability = readability_reward(
        completions=completions,
        test=kwargs["test"],
        entry_point=kwargs["entry_point"],
        prompt=kwargs["prompt"],
        timeout=timeout,
    )
    efficiency = efficiency_reward(
        completions=completions,
        test=kwargs["test"],
        entry_point=kwargs["entry_point"],
        prompt=kwargs["prompt"],
        canonical_solution=kwargs["canonical_solution"],
        timeout=timeout,
    )

    total = [
        reward_weights[0] * c + reward_weights[1] * r + reward_weights[2] * e
        for c, r, e in zip(correctness, readability, efficiency)
    ]

    samples = []
    for idx, completion in enumerate(completions):
        content = completion[0]["content"]
        code = extract_code_from_completion(content)
        preview = code.strip().splitlines()
        preview_text = "\n".join(preview[:8]).strip()
        samples.append(
            {
                "sample_id": idx,
                "correctness": correctness[idx],
                "readability": readability[idx],
                "efficiency": efficiency[idx],
                "total_reward": total[idx],
                "preview": preview_text[:600],
            }
        )

    return {
        "task_id": row["task_id"],
        "entry_point": row["entry_point"],
        "samples": samples,
    }


def _group_stats(values: list[float]) -> dict:
    """Summarize reward spread inside one prompt group."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "range": 0.0, "unique": 0}
    return {
        "mean": sum(values) / len(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "range": max(values) - min(values),
        "unique": len({round(v, 6) for v in values}),
    }


def _summarize_results(results: list[dict], discrimination_threshold: float) -> dict:
    """Aggregate prompt-level reward discrimination diagnostics."""
    reward_names = ["correctness", "readability", "efficiency", "total_reward"]
    summary = {}

    for reward_name in reward_names:
        group_stats = [
            _group_stats([sample[reward_name] for sample in item["samples"]])
            for item in results
        ]
        summary[reward_name] = {
            "avg_mean": sum(s["mean"] for s in group_stats) / max(len(group_stats), 1),
            "avg_std": sum(s["std"] for s in group_stats) / max(len(group_stats), 1),
            "avg_range": sum(s["range"] for s in group_stats) / max(len(group_stats), 1),
            "avg_unique": sum(s["unique"] for s in group_stats) / max(len(group_stats), 1),
            "discriminative_prompt_ratio": (
                sum(1 for s in group_stats if s["range"] > discrimination_threshold)
                / max(len(group_stats), 1)
            ),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Check whether GRPO rewards can discriminate sampled completions."
    )
    parser.add_argument(
        "--config",
        default="configs/grpo_config.yaml",
        help="Path to the GRPO config file.",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=16,
        help="How many prompts to sample from the GRPO dataset.",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="How many completions to sample per prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling prompts.",
    )
    parser.add_argument(
        "--output_file",
        default="outputs/grpo/reward_check.json",
        help="Where to save detailed reward-check results.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Override max completion length from config.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override sampling temperature from config.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Override top-p from config.",
    )
    parser.add_argument(
        "--discrimination_threshold",
        type=float,
        default=0.05,
        help="Minimum within-prompt reward range considered discriminative.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    model_path = config["model"]["name"]
    grpo_cfg = config["grpo"]
    data_cfg = config["data"]
    sandbox_cfg = config.get("sandbox", {})

    dataset = load_grpo_dataset(data_cfg["prompt_file"])
    rows = list(dataset)
    if args.num_prompts < len(rows):
        rows = random.sample(rows, args.num_prompts)

    num_generations = args.num_generations or grpo_cfg.get("num_generations", 8)
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else grpo_cfg.get("max_completion_length", 1024)
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else grpo_cfg.get("temperature", 0.7)
    )
    top_p = args.top_p if args.top_p is not None else grpo_cfg.get("top_p", 0.95)
    timeout = sandbox_cfg.get("timeout", 10)
    reward_weights = grpo_cfg.get("reward_weights", [0.70, 0.15, 0.15])

    print("=" * 60)
    print("GRPO Reward Sanity Check")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Prompt file: {data_cfg['prompt_file']}")
    print(f"Checked prompts: {len(rows)}")
    print(f"Generations per prompt: {num_generations}")

    model, tokenizer = load_model_for_inference(model_path)
    model.eval()

    results = []
    for idx, row in enumerate(rows, start=1):
        prompt_messages = _extract_prompt_batch(row["prompt"])
        print(f"[{idx}/{len(rows)}] {row['task_id']}")
        completions = _generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompt_messages=prompt_messages,
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        results.append(
            _score_prompt_group(
                row=row,
                completions=completions,
                reward_weights=reward_weights,
                timeout=timeout,
            )
        )

    summary = _summarize_results(
        results=results,
        discrimination_threshold=args.discrimination_threshold,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": args.config,
        "model_path": model_path,
        "num_prompts": len(rows),
        "num_generations": num_generations,
        "discrimination_threshold": args.discrimination_threshold,
        "summary": summary,
        "results": results,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\nReward discrimination summary:")
    for reward_name, stats in summary.items():
        print(
            f"- {reward_name}: "
            f"avg_std={stats['avg_std']:.4f}, "
            f"avg_range={stats['avg_range']:.4f}, "
            f"avg_unique={stats['avg_unique']:.2f}, "
            f"discriminative_prompt_ratio={stats['discriminative_prompt_ratio']:.2%}"
        )
    print(f"\nSaved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
