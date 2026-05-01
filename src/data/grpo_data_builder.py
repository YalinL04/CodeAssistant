"""Build GRPO prompt datasets for RL training."""

import json
import re
import textwrap
from pathlib import Path

from datasets import Dataset

from src.data.dataset_loader import load_humaneval, load_mbpp
from src.data.prompts import SYSTEM_PROMPT, format_code_prompt


def _extract_function_signature(code: str) -> str:
    """Extract the first Python function signature from a code snippet."""
    match = re.search(r"(def\s+\w+\s*\([^)]*\)\s*(?:->.*?)?:)", code)
    if not match:
        raise ValueError("Could not extract function signature from code")
    return match.group(1)


def _extract_entry_point(code: str) -> str:
    """Extract the first Python function name from a code snippet."""
    match = re.search(r"def\s+(\w+)\s*\(", code)
    if not match:
        raise ValueError("Could not extract entry point from code")
    return match.group(1)


def _build_mbpp_prompt_code(problem_text: str, code: str) -> str:
    """Build the prompt-side function stub shown to the model for MBPP."""
    signature = _extract_function_signature(code)
    docstring = textwrap.indent(f'"""{problem_text.strip()}"""', "    ")
    return f"{signature}\n{docstring}\n"


def _build_mbpp_test_harness(
    test_list: list[str],
    entry_point: str,
    test_setup_code: str = "",
) -> str:
    """
    Convert MBPP asserts into the `check(candidate)` contract used by rewards.

    MBPP tests usually call the original function name directly. We alias that
    name to `candidate` inside the harness so the existing sandbox executor can
    keep calling `check(entry_point)`.
    """
    lines = []
    if test_setup_code and test_setup_code.strip():
        lines.append(test_setup_code.rstrip())
        lines.append("")

    lines.append("def check(candidate):")
    lines.append(f"    {entry_point} = candidate")
    for raw_assert in test_list:
        stripped = raw_assert.strip()
        if stripped:
            lines.append(f"    {stripped}")

    return "\n".join(lines)


def build_humaneval_grpo_prompt(task_id: str, problem: dict) -> dict:
    """Build a single GRPO prompt row from a HumanEval problem."""
    user_content = format_code_prompt(problem["prompt"])

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "test": problem["test"],
        "entry_point": problem["entry_point"],
        "task_id": task_id,
        "canonical_solution": problem["canonical_solution"],
    }


def build_mbpp_grpo_prompt(sample: dict) -> dict:
    """Build a single GRPO prompt row from an MBPP sample."""
    prompt_code = _build_mbpp_prompt_code(sample["text"], sample["code"])
    entry_point = _extract_entry_point(sample["code"])
    test_code = _build_mbpp_test_harness(
        test_list=sample["test_list"],
        entry_point=entry_point,
        test_setup_code=sample.get("test_setup_code", ""),
    )

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_code_prompt(prompt_code)},
        ],
        "test": test_code,
        "entry_point": entry_point,
        "task_id": f"MBPP/{sample['task_id']}",
        "canonical_solution": sample["code"],
    }


def build_grpo_dataset(
    output_dir: str = "data/processed",
    dataset_name: str = "humaneval",
    split: str = "train",
    output_file: str = "grpo_prompts.jsonl",
) -> Dataset:
    """
    Build a GRPO prompt dataset and save it as JSONL.

    Returns a HuggingFace Dataset and optionally saves to JSONL.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = []
    if dataset_name == "humaneval":
        problems = load_humaneval()
        for task_id, problem in problems.items():
            rows.append(build_humaneval_grpo_prompt(task_id, problem))
    elif dataset_name == "mbpp":
        problems = load_mbpp(split=split)
        for sample in problems:
            rows.append(build_mbpp_grpo_prompt(sample))
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    # Save to JSONL for reproducibility
    jsonl_file = output_path / output_file
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Built {len(rows)} GRPO prompts from {dataset_name}:{split} -> {jsonl_file}")

    dataset = Dataset.from_list(rows)
    return dataset


if __name__ == "__main__":
    build_grpo_dataset()
