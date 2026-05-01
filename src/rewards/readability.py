"""Code readability reward: static analysis based scoring."""

import re

from src.rewards.correctness import compute_correctness_scores
from src.utils.code_utils import (
    extract_code_from_completion,
    get_variable_names,
    has_docstring,
    count_nesting_depth,
)

# Common acceptable single-letter variable names (loop vars, math conventions)
ACCEPTABLE_SHORT_NAMES = {"i", "j", "k", "n", "m", "x", "y", "z", "v", "e", "f", "s", "c", "d", "p", "q", "r", "t", "_"}


def score_variable_naming(code: str) -> float:
    """
    Score variable naming quality.

    Penalizes single-character names (except common loop/math vars).
    Rewards descriptive snake_case names.
    Returns score in [0.0, 1.0].
    """
    names = get_variable_names(code)
    if not names:
        return 0.5  # neutral if no variables

    good_count = 0
    for name in names:
        if name.startswith("_"):
            good_count += 1  # private/unused convention
        elif len(name) == 1 and name not in ACCEPTABLE_SHORT_NAMES:
            continue  # bad single-char name
        elif len(name) >= 2:
            good_count += 1  # descriptive enough
        elif name in ACCEPTABLE_SHORT_NAMES:
            good_count += 0.5  # acceptable but not great

    return min(1.0, good_count / max(len(names), 1))


def score_documentation(code: str) -> float:
    """
    Score documentation quality.

    Checks for docstrings and inline comments.
    Returns score in [0.0, 1.0].
    """
    score = 0.0

    if has_docstring(code):
        score += 0.6

    # Check for inline comments
    lines = code.strip().split("\n")
    comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
    if comment_lines > 0:
        score += min(0.4, comment_lines * 0.1)

    return min(1.0, score)


def score_line_length(code: str, max_length: int = 100) -> float:
    """
    Score line length compliance.

    Returns score in [0.0, 1.0] based on fraction of lines within limit.
    """
    lines = code.strip().split("\n")
    if not lines:
        return 1.0

    good_lines = sum(1 for line in lines if len(line) <= max_length)
    return good_lines / len(lines)


def score_nesting(code: str, max_acceptable: int = 4) -> float:
    """
    Score based on nesting depth.

    Deep nesting reduces readability. Returns 1.0 for shallow code,
    decreasing as nesting increases.
    """
    depth = count_nesting_depth(code)
    if depth <= 2:
        return 1.0
    elif depth <= max_acceptable:
        return 1.0 - (depth - 2) / (max_acceptable - 2) * 0.5
    else:
        return max(0.0, 0.5 - (depth - max_acceptable) * 0.1)


def score_reasoning_trace(completion: str) -> float:
    """
    Score whether the completion includes a reasoning trace.

    Returns 1.0 if <think>...</think> is present and non-empty,
    0.0 otherwise.
    """
    match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
    if match and len(match.group(1).strip()) > 20:
        return 1.0
    return 0.0


def readability_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    """
    Compute readability reward based on static code analysis.

    Sub-signals and their weights:
    - Variable naming: 0.25
    - Documentation: 0.20
    - Line length: 0.20
    - Nesting depth: 0.20
    - Reasoning trace: 0.15

    Args:
        completions: List of completions in conversation format

    Returns:
        List of float rewards in [0.0, 1.0]
    """
    base_rewards = []

    for completion in completions:
        # Extract text content
        if isinstance(completion, list):
            content = ""
            for msg in completion:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
        elif isinstance(completion, dict):
            content = completion.get("content", "")
        else:
            content = str(completion)

        code = extract_code_from_completion(content)

        if not code.strip():
            base_rewards.append(0.0)
            continue

        # Compute sub-scores
        naming_score = score_variable_naming(code)
        doc_score = score_documentation(code)
        length_score = score_line_length(code)
        nesting_score = score_nesting(code)
        #reasoning_score = score_reasoning_trace(content)

        # Weighted combination
        total = (
            0.25 * naming_score
            + 0.25 * doc_score
            + 0.25 * length_score
            + 0.25 * nesting_score
            #+ 0.15 * reasoning_score
        )

        base_rewards.append(total)

    test = kwargs.get("test")
    entry_point = kwargs.get("entry_point")
    if test is None or entry_point is None:
        return base_rewards

    correctness_scores = compute_correctness_scores(
        completions=completions,
        test=test,
        entry_point=entry_point,
        prompt=kwargs.get("prompt"),
        timeout=kwargs.get("timeout", 10),
    )
    return [
        base_reward * correctness
        for base_reward, correctness in zip(base_rewards, correctness_scores)
    ]
