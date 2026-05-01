"""Code efficiency reward: complexity heuristics."""

import math
import time
import textwrap

from src.rewards.correctness import (
    build_executable_solution_from_completion,
    compute_correctness_scores,
)
from src.rewards.sandbox import execute_with_tests
from src.utils.code_utils import (
    extract_code_from_completion,
    count_loop_nesting,
    uses_efficient_structures,
)


def estimate_complexity_score(code: str) -> float:
    """
    Estimate algorithmic complexity and return a score.

    Lower complexity -> higher score.
    Score in [0.0, 1.0].
    """
    loop_depth = count_loop_nesting(code)

    if loop_depth == 0:
        return 1.0   # O(1) or single-pass
    elif loop_depth == 1:
        return 0.8   # O(n) - good
    elif loop_depth == 2:
        return 0.5   # O(n^2) - acceptable
    elif loop_depth == 3:
        return 0.3   # O(n^3) - poor
    else:
        return 0.1   # O(n^4+) - very poor


def score_efficient_constructs(code: str) -> float:
    """
    Reward usage of efficient Python constructs.

    Returns score in [0.0, 1.0] based on how many efficient
    patterns are used.
    """
    indicators = uses_efficient_structures(code)

    efficient_count = 0
    total_checks = 0

    # Efficient lookup structures
    if indicators["set"] or indicators["dict"] or indicators["defaultdict"] or indicators["Counter"]:
        efficient_count += 1
    total_checks += 1

    # Efficient iteration
    if indicators["enumerate"] or indicators["zip"]:
        efficient_count += 1
    total_checks += 1

    # Pythonic constructs
    if indicators["list_comprehension"]:
        efficient_count += 1
    total_checks += 1

    # Sorting (sometimes indicates O(n log n) instead of O(n^2))
    if indicators["sorted"]:
        efficient_count += 0.5
    total_checks += 1

    if total_checks == 0:
        return 0.5

    return min(1.0, efficient_count / total_checks + 0.3)


def compare_with_canonical(generated_code: str, canonical_code: str) -> float:
    """
    Compare generated code's complexity with canonical solution.

    Returns:
        1.0 if generated is at least as efficient
        0.5 if similar complexity
        0.0 if significantly worse
    """
    gen_depth = count_loop_nesting(generated_code)
    can_depth = count_loop_nesting(canonical_code)

    if gen_depth <= can_depth:
        return 1.0   # at least as good
    elif gen_depth == can_depth + 1:
        return 0.5   # slightly worse
    else:
        return 0.2   # significantly worse


def _measure_test_runtime(
    solution: str,
    test_code: str,
    entry_point: str,
    timeout: int,
) -> float | None:
    """
    Measure wall-clock runtime of executing a solution in the shared sandbox.

    Returns None when execution fails or times out.
    """
    if not solution.strip():
        return None

    start = time.perf_counter()
    result = execute_with_tests(
        code=solution,
        test_code=test_code,
        entry_point=entry_point,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start

    if not result.success:
        return None
    return elapsed


def _relative_runtime_score(
    generated_runtime: float | None,
    canonical_runtime: float | None,
) -> float:
    """
    Convert a generated/canonical runtime ratio into a bounded reward.

    Matching the canonical runtime receives 1.0. Faster code saturates at 1.0.
    Slower code decays smoothly toward 0.0 on a log scale.
    """
    if generated_runtime is None or canonical_runtime is None:
        return 0.0

    canonical_runtime = max(canonical_runtime, 1e-6)
    generated_runtime = max(generated_runtime, 1e-6)

    ratio = generated_runtime / canonical_runtime
    if ratio <= 1.0:
        return 1.0

    score = 1.0 - math.log(ratio) / math.log(4.0)
    return max(0.0, min(1.0, score))


def efficiency_reward(completions: list[list[dict]],
                      canonical_solution: list[str] = None,
                      **kwargs) -> list[float]:
    """
    Compute efficiency reward based on complexity heuristics.

    Sub-signals:
    - Absolute complexity score: 0.35
    - Efficient construct usage: 0.30
    - Comparison with canonical: 0.35

    Args:
        completions: List of completions in conversation format
        canonical_solution: List of canonical solutions for comparison

    Returns:
        List of float rewards in [0.0, 1.0]
    """
    if canonical_solution is None:
        canonical_solution = [None] * len(completions)

    test = kwargs.get("test")
    entry_point = kwargs.get("entry_point")
    prompt = kwargs.get("prompt")
    timeout = kwargs.get("timeout", 10)

    if test is None or entry_point is None:
        return [0.0] * len(completions)

    correctness_scores = compute_correctness_scores(
        completions=completions,
        test=test,
        entry_point=entry_point,
        prompt=prompt,
        timeout=timeout,
    )

    if prompt is None:
        prompt = [None] * len(completions)

    base_rewards = []

    for completion, canonical, correctness, test_code, func_name, prompt_messages in zip(
        completions,
        canonical_solution,
        correctness_scores,
        test,
        entry_point,
        prompt,
    ):
        if correctness != 1.0:
            base_rewards.append(0.0)
            continue

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

        # Sub-scores
        complexity_score = estimate_complexity_score(code)
        construct_score = score_efficient_constructs(code)

        generated_solution = build_executable_solution_from_completion(
            completion=completion,
            prompt_messages=prompt_messages,
        )
        canonical_solution_code = ""
        if canonical and canonical.strip():
            canonical_solution_code = build_executable_solution_from_completion(
                completion={"content": canonical},
                prompt_messages=prompt_messages,
            )

        generated_runtime = _measure_test_runtime(
            solution=generated_solution,
            test_code=test_code,
            entry_point=func_name,
            timeout=timeout,
        )
        canonical_runtime = _measure_test_runtime(
            solution=canonical_solution_code,
            test_code=test_code,
            entry_point=func_name,
            timeout=timeout,
        )
        relative_runtime_score = _relative_runtime_score(
            generated_runtime=generated_runtime,
            canonical_runtime=canonical_runtime,
        )

        # Weighted combination
        total = (
            0.35 * complexity_score
            + 0.30 * construct_score
            + 0.35 * relative_runtime_score
        )

        base_rewards.append(total)
    return base_rewards
