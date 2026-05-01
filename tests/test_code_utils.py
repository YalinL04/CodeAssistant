"""Tests for code utility functions."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.code_utils import (
    extract_code_from_completion,
    count_loop_nesting,
    count_nesting_depth,
    get_variable_names,
    has_docstring,
    uses_efficient_structures,
    strip_prompt_prefix,
)


class TestExtractCode(unittest.TestCase):
    def test_extract_from_think_and_code_block(self):
        completion = """<think>
Let me think about this problem.
I need to iterate through the list.
</think>

```python
def add(a, b):
    return a + b
```"""
        code = extract_code_from_completion(completion)
        self.assertIn("def add(a, b):", code)
        self.assertNotIn("<think>", code)

    def test_extract_code_block_only(self):
        completion = """```python
def multiply(x, y):
    return x * y
```"""
        code = extract_code_from_completion(completion)
        self.assertIn("def multiply(x, y):", code)

    def test_extract_plain_code(self):
        completion = "def hello():\n    return 'world'"
        code = extract_code_from_completion(completion)
        self.assertIn("def hello():", code)

    def test_empty_completion(self):
        code = extract_code_from_completion("")
        self.assertEqual(code, "")


class TestLoopNesting(unittest.TestCase):
    def test_no_loops(self):
        code = "x = 1\ny = 2\nreturn x + y"
        self.assertEqual(count_loop_nesting(code), 0)

    def test_single_loop(self):
        code = "for i in range(10):\n    print(i)"
        self.assertEqual(count_loop_nesting(code), 1)

    def test_nested_loops(self):
        code = """
for i in range(10):
    for j in range(10):
        print(i, j)
"""
        self.assertEqual(count_loop_nesting(code), 2)

    def test_triple_nested(self):
        code = """
for i in range(5):
    for j in range(5):
        for k in range(5):
            pass
"""
        self.assertEqual(count_loop_nesting(code), 3)


class TestVariableNames(unittest.TestCase):
    def test_basic_names(self):
        code = "result = 0\ncount = 10"
        names = get_variable_names(code)
        self.assertIn("result", names)
        self.assertIn("count", names)

    def test_function_args(self):
        code = "def foo(numbers, threshold):\n    pass"
        names = get_variable_names(code)
        self.assertIn("numbers", names)
        self.assertIn("threshold", names)


class TestDocstring(unittest.TestCase):
    def test_has_docstring(self):
        code = 'def foo():\n    """A function."""\n    pass'
        self.assertTrue(has_docstring(code))

    def test_no_docstring(self):
        code = "def foo():\n    pass"
        self.assertFalse(has_docstring(code))


class TestEfficientStructures(unittest.TestCase):
    def test_set_usage(self):
        code = "s = set(items)\nif x in s: pass"
        indicators = uses_efficient_structures(code)
        self.assertTrue(indicators["set"])

    def test_comprehension(self):
        code = "result = [x*2 for x in range(10)]"
        indicators = uses_efficient_structures(code)
        self.assertTrue(indicators["list_comprehension"])

    def test_enumerate(self):
        code = "for i, v in enumerate(lst): pass"
        indicators = uses_efficient_structures(code)
        self.assertTrue(indicators["enumerate"])


class TestStripPromptPrefix(unittest.TestCase):
    def test_with_prefix(self):
        prompt = "def add(a, b):\n"
        generated = "def add(a, b):\n    return a + b"
        result = strip_prompt_prefix(generated, prompt)
        self.assertEqual(result.strip(), "return a + b")

    def test_without_prefix(self):
        prompt = "def add(a, b):\n"
        generated = "    return a + b"
        result = strip_prompt_prefix(generated, prompt)
        self.assertEqual(result.strip(), "return a + b")


if __name__ == "__main__":
    unittest.main()
