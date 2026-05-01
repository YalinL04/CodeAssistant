"""Tests for reward functions."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rewards.correctness import correctness_reward
from src.rewards.readability import (
    readability_reward,
    score_variable_naming,
    score_documentation,
    score_line_length,
    score_nesting,
    score_reasoning_trace,
)
from src.rewards.efficiency import (
    efficiency_reward,
    estimate_complexity_score,
    score_efficient_constructs,
    compare_with_canonical,
)


def make_completion(content: str) -> list[dict]:
    """Helper to create a completion in conversation format."""
    return [{"role": "assistant", "content": content}]


class TestCorrectnessReward(unittest.TestCase):
    def test_correct_code(self):
        completion = make_completion("""<think>
Simple addition.
</think>

```python
def add(a, b):
    return a + b
```""")
        test_code = """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(-1, 1) == 0
"""
        rewards = correctness_reward(
            completions=[completion],
            test=["" + test_code],
            entry_point=["add"],
        )
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_incorrect_code(self):
        completion = make_completion("""```python
def add(a, b):
    return a - b
```""")
        test_code = """
def check(candidate):
    assert candidate(1, 2) == 3
"""
        rewards = correctness_reward(
            completions=[completion],
            test=["" + test_code],
            entry_point=["add"],
        )
        self.assertEqual(rewards[0], 0.0)

    def test_partial_credit(self):
        completion = make_completion("""```python
def add(a, b):
    if a == 1 and b == 2:
        return 3
    return -1
```""")
        test_code = """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(2, 3) == 5
"""
        rewards = correctness_reward(
            completions=[completion],
            test=["" + test_code],
            entry_point=["add"],
        )
        self.assertEqual(rewards[0], 0.5)

    def test_empty_completion(self):
        completion = make_completion("")
        rewards = correctness_reward(
            completions=[completion],
            test=["def check(c): pass"],
            entry_point=["foo"],
        )
        self.assertEqual(rewards[0], 0.0)


class TestReadabilitySubScores(unittest.TestCase):
    def test_good_naming(self):
        code = "result = 0\ntotal_count = 10\nmax_value = 100"
        score = score_variable_naming(code)
        self.assertGreater(score, 0.5)

    def test_bad_naming(self):
        code = "a = 0\nb = 10\nc = 100\nd = a + b"
        score = score_variable_naming(code)
        self.assertLess(score, 0.8)

    def test_documentation_with_docstring(self):
        code = 'def foo():\n    """Does something."""\n    pass'
        score = score_documentation(code)
        self.assertGreaterEqual(score, 0.6)

    def test_documentation_without(self):
        code = "def foo():\n    pass"
        score = score_documentation(code)
        self.assertEqual(score, 0.0)

    def test_line_length_good(self):
        code = "x = 1\ny = 2\nz = x + y"
        score = score_line_length(code)
        self.assertEqual(score, 1.0)

    def test_nesting_shallow(self):
        code = "if True:\n    pass"
        score = score_nesting(code)
        self.assertEqual(score, 1.0)

    def test_reasoning_trace(self):
        text = "<think>\nLet me analyze this problem carefully.\n</think>\ncode here"
        score = score_reasoning_trace(text)
        self.assertEqual(score, 1.0)

    def test_no_reasoning_trace(self):
        text = "def foo(): pass"
        score = score_reasoning_trace(text)
        self.assertEqual(score, 0.0)


class TestReadabilityReward(unittest.TestCase):
    def test_good_code(self):
        completion = make_completion("""<think>
Let me analyze this problem carefully and think step by step.
</think>

```python
def calculate_sum(numbers):
    \"\"\"Calculate the sum of a list of numbers.\"\"\"
    total = 0
    for num in numbers:
        total += num
    return total
```""")
        rewards = readability_reward(completions=[completion])
        self.assertEqual(len(rewards), 1)
        self.assertGreater(rewards[0], 0.5)

    def test_readability_is_gated_by_correctness(self):
        completion = make_completion("""<think>
Let me analyze this carefully and write clean code.
</think>

```python
def add(a, b):
    \"\"\"Add two numbers.\"\"\"
    total = a - b
    return total
```""")
        test_code = """
def check(candidate):
    assert candidate(1, 2) == 3
"""
        rewards = readability_reward(
            completions=[completion],
            test=["" + test_code],
            entry_point=["add"],
        )
        self.assertEqual(rewards[0], 0.0)


class TestEfficiencySubScores(unittest.TestCase):
    def test_constant_complexity(self):
        code = "return a + b"
        score = estimate_complexity_score(code)
        self.assertEqual(score, 1.0)

    def test_linear_complexity(self):
        code = "for x in items:\n    pass"
        score = estimate_complexity_score(code)
        self.assertEqual(score, 0.8)

    def test_quadratic_complexity(self):
        code = "for i in items:\n    for j in items:\n        pass"
        score = estimate_complexity_score(code)
        self.assertEqual(score, 0.5)

    def test_efficient_constructs(self):
        code = "seen = set(items)\nresult = [x for x in data if x in seen]"
        score = score_efficient_constructs(code)
        self.assertGreater(score, 0.5)

    def test_canonical_comparison_equal(self):
        gen = "for x in items:\n    pass"
        can = "for x in items:\n    pass"
        score = compare_with_canonical(gen, can)
        self.assertEqual(score, 1.0)

    def test_canonical_comparison_worse(self):
        gen = "for i in items:\n    for j in items:\n        pass"
        can = "for x in items:\n    pass"
        score = compare_with_canonical(gen, can)
        self.assertLess(score, 1.0)


class TestEfficiencyReward(unittest.TestCase):
    def test_efficient_code(self):
        completion = make_completion("""```python
def find_unique(items):
    return list(set(items))
```""")
        rewards = efficiency_reward(
            completions=[completion],
            canonical_solution=["return list(set(items))"],
        )
        self.assertEqual(len(rewards), 1)
        self.assertGreater(rewards[0], 0.5)

    def test_efficiency_is_scaled_by_partial_correctness(self):
        completion = make_completion("""```python
def add(a, b):
    if a == 1 and b == 2:
        return 3
    return a - b
```""")
        test_code = """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(2, 3) == 5
"""
        ungated = efficiency_reward(
            completions=[completion],
            canonical_solution=["def add(a, b):\n    return a + b"],
        )[0]
        gated = efficiency_reward(
            completions=[completion],
            canonical_solution=["def add(a, b):\n    return a + b"],
            test=["" + test_code],
            entry_point=["add"],
        )[0]
        self.assertGreater(ungated, gated)
        self.assertAlmostEqual(gated, ungated * 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
