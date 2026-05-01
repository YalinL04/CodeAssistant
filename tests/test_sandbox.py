"""Tests for the code execution sandbox."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rewards.sandbox import sandbox_execute, execute_with_tests


class TestSandboxExecute(unittest.TestCase):
    def test_simple_success(self):
        code = "print('hello')"
        result = sandbox_execute(code)
        self.assertTrue(result.success)
        self.assertIn("hello", result.stdout)

    def test_syntax_error(self):
        code = "def foo(\n"
        result = sandbox_execute(code)
        self.assertFalse(result.success)

    def test_runtime_error(self):
        code = "x = 1 / 0"
        result = sandbox_execute(code)
        self.assertFalse(result.success)
        self.assertIn("ZeroDivision", result.stderr)

    def test_timeout(self):
        code = "while True: pass"
        result = sandbox_execute(code, timeout=2)
        self.assertFalse(result.success)
        self.assertTrue(result.timed_out)

    def test_blocked_import(self):
        code = "import subprocess"
        result = sandbox_execute(code)
        self.assertFalse(result.success)

    def test_computation(self):
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

assert fibonacci(10) == 55
print("pass")
"""
        result = sandbox_execute(code)
        self.assertTrue(result.success)
        self.assertIn("pass", result.stdout)


class TestExecuteWithTests(unittest.TestCase):
    def test_correct_solution(self):
        code = """
def add(a, b):
    return a + b
"""
        test_code = """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(-1, 1) == 0
    assert candidate(0, 0) == 0
"""
        result = execute_with_tests(code, test_code, "add")
        self.assertTrue(result.success)

    def test_wrong_solution(self):
        code = """
def add(a, b):
    return a - b
"""
        test_code = """
def check(candidate):
    assert candidate(1, 2) == 3
"""
        result = execute_with_tests(code, test_code, "add")
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()
