"""Code correctness reward: execute against test cases."""

import re
from typing import Any

from src.utils.code_utils import extract_code_from_completion
from src.rewards.sandbox import execute_with_tests

_LAST_CORRECTNESS_CACHE: dict[str, Any] = {
    "key": None,
    "scores": None,
}


def _extract_prompt_code(prompt_messages) -> str:
    """
    Recover the raw HumanEval prompt code from the GRPO chat prompt.

    The user message is formatted as:
    Complete the following Python function:

    ```python
    ...
    ```
    """
    if not isinstance(prompt_messages, list):
        return ""

    for msg in prompt_messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        match = re.search(r"```python\n(.*?)\n```", content, flags=re.DOTALL)
        if match:
            return match.group(1)
    return ""


def _build_executable_solution(prompt_code: str, generated_code: str) -> str:
    """
    Build a full executable solution for HumanEval-style tasks.

    GRPO generations may contain:
    - only the function body
    - a full regenerated function
    - imports + function definition
    """
    prompt_stripped = prompt_code.strip()
    generated_stripped = generated_code.strip()

    if not generated_stripped:
        return ""

    if not prompt_stripped:
        return generated_code

    if generated_stripped.startswith(prompt_stripped):
        return generated_code

    if re.search(r"^\s*def\s+", generated_code) or re.search(
        r"^\s*(from\s+\S+\s+import|import\s+\S+)", generated_code
    ):
        return generated_code

    return prompt_code + generated_code


def _extract_completion_content(completion: list[dict] | dict | str) -> str:
    """Extract assistant content from a GRPO completion payload."""
    if isinstance(completion, list):
        content = ""
        for msg in completion:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
        return content
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def build_executable_solution_from_completion(
    completion: list[dict] | dict | str,
    prompt_messages: list[dict] | None = None,
) -> str:
    """
    Build an executable Python solution from a GRPO completion payload.

    This reuses the same prompt-recovery and code-extraction path as the
    correctness reward so other rewards can evaluate the exact same artifact.
    """
    content = _extract_completion_content(completion)
    code = extract_code_from_completion(content)
    prompt_code = _extract_prompt_code(prompt_messages)
    return _build_executable_solution(prompt_code, code)


def _split_test_code(test_code: str) -> tuple[list[str], list[list[str]]]:
    """
    Split a `check(candidate)` harness into shared setup and per-assert blocks.

    Each block contains the statements needed before a single assert so we can
    award partial credit instead of a strict all-or-nothing score.
    """
    setup_lines = []
    assert_blocks = []
    in_check = False
    current_prefix = []

    for raw_line in test_code.splitlines():
        if raw_line.startswith("def check("):
            in_check = True
            continue

        if not in_check:
            setup_lines.append(raw_line)
            continue

        if not raw_line.strip():
            continue

        if not raw_line.startswith(("    ", "\t")):
            setup_lines.append(raw_line)
            continue

        stripped = raw_line.strip()
        if stripped.startswith("assert "):
            assert_blocks.append(current_prefix + [stripped])
        else:
            current_prefix.append(stripped)

    return setup_lines, assert_blocks


def _score_partial_correctness(solution: str, test_code: str, entry_point: str, timeout: int) -> float:
    """Run tests and return the pass fraction when full execution fails."""
    setup_lines, assert_blocks = _split_test_code(test_code)
    if not assert_blocks:
        return 0.0

    passed = 0
    for block in assert_blocks:
        result = execute_with_tests(
            code=solution,
            test_code="\n".join(
                line for line in (
                    "\n".join(setup_lines).strip(),
                    "",
                    "def check(candidate):",
                    *[f"    {line}" for line in block],
                )
                if line != ""
            ),
            entry_point=entry_point,
            timeout=timeout,
        )
        if result.success:
            passed += 1

    return passed / len(assert_blocks)


def compute_correctness_scores(
    completions: list[list[dict]],
    test: list[str],
    entry_point: list[str],
    prompt: list[list[dict]] = None,
    timeout: int = 10,
) -> list[float]:
    """
    Compute dense correctness scores and cache the latest batch result.

    Full pass receives 1.0. Otherwise we rerun tests one assert at a time and
    return the fraction passed, which makes the reward much less sparse.
    """
    cache_key = (
        tuple(_extract_completion_content(completion) for completion in completions),
        tuple(test),
        tuple(entry_point),
        tuple(str(item) for item in prompt) if prompt is not None else None,
        timeout,
    )
    if _LAST_CORRECTNESS_CACHE["key"] == cache_key:
        return list(_LAST_CORRECTNESS_CACHE["scores"])

    if prompt is None:
        prompt = [None] * len(completions)
    rewards = []

    for completion, test_code, func_name, prompt_messages in zip(
        completions, test, entry_point, prompt
    ):
        solution = build_executable_solution_from_completion(
            completion=completion,
            prompt_messages=prompt_messages,
        )

        if not solution.strip():
            rewards.append(0.0)
            continue

        # Execute against test cases
        result = execute_with_tests(
            code=solution,
            test_code=test_code,
            entry_point=func_name,
            timeout=timeout,
        )

        if result.success:
            rewards.append(1.0)
        else:
            rewards.append(
                _score_partial_correctness(
                    solution=solution,
                    test_code=test_code,
                    entry_point=func_name,
                    timeout=timeout,
                )
            )

    _LAST_CORRECTNESS_CACHE["key"] = cache_key
    _LAST_CORRECTNESS_CACHE["scores"] = list(rewards)
    return rewards


def correctness_reward(completions: list[list[dict]], test: list[str],
                       entry_point: list[str], prompt: list[list[dict]] = None,
                       **kwargs) -> list[float]:
    """Public reward entry point used by GRPOTrainer."""
    return compute_correctness_scores(
        completions=completions,
        test=test,
        entry_point=entry_point,
        prompt=prompt,
        timeout=kwargs.get("timeout", 10),
    )
