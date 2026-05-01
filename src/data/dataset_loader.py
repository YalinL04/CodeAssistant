"""Load SFT, RL, and evaluation datasets."""

from datasets import load_dataset, Dataset


def load_mbpp(split: str = "train") -> Dataset:
    """
    Load the MBPP (Mostly Basic Python Problems) dataset.

    Splits available: 'train' (~374), 'validation' (~90), 'test' (~500)
    Each sample has: task_id, text (problem description), code (solution),
                     test_list (list of assert statements), test_setup_code
    """
    # ds = load_dataset("mbpp", "sanitized", split=split, trust_remote_code=True)
    # 方式 A：加载完整版 (包含 train, test, validation, prompt 等分片)
    ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
    return ds


def load_codealpaca(split: str = "train") -> Dataset:
    """
    Load the CodeAlpaca-20k dataset for instruction tuning.

    The dataset is hosted on Hugging Face and typically contains a single
    train split with columns like: instruction, input, output.
    """
    return load_dataset("sahil2801/CodeAlpaca-20k", split=split)


def load_humaneval() -> dict:
    """
    Load HumanEval problems via evalplus for GRPO and evaluation.

    Returns dict: task_id -> {prompt, canonical_solution, test, entry_point, ...}
    """
    from evalplus.data import get_human_eval_plus
    problems = get_human_eval_plus()
    return problems


def load_humaneval_as_dataset() -> Dataset:
    """Load HumanEval as a HuggingFace Dataset for GRPO training."""
    problems = load_humaneval()
    rows = []
    for task_id, problem in problems.items():
        rows.append({
            "task_id": task_id,
            "prompt": problem["prompt"],
            "canonical_solution": problem["canonical_solution"],
            "test": problem["test"],
            "entry_point": problem["entry_point"],
        })
    return Dataset.from_list(rows)
