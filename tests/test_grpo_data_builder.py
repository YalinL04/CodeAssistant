"""Tests for GRPO dataset building."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.grpo_data_builder import (
    build_mbpp_grpo_prompt,
    _build_mbpp_test_harness,
)


class TestMbppGrpoBuilder(unittest.TestCase):
    def test_build_mbpp_test_harness_aliases_candidate(self):
        test_code = _build_mbpp_test_harness(
            test_list=[
                "assert add(1, 2) == 3",
                "assert add(0, 0) == 0",
            ],
            entry_point="add",
            test_setup_code="import math",
        )
        self.assertIn("import math", test_code)
        self.assertIn("def check(candidate):", test_code)
        self.assertIn("add = candidate", test_code)
        self.assertIn("assert add(1, 2) == 3", test_code)

    def test_build_mbpp_grpo_prompt(self):
        sample = {
            "task_id": 1,
            "text": "Write a function to add two numbers.",
            "code": "def add(a, b):\n    return a + b\n",
            "test_list": ["assert add(1, 2) == 3"],
            "test_setup_code": "",
        }

        row = build_mbpp_grpo_prompt(sample)

        self.assertEqual(row["task_id"], "MBPP/1")
        self.assertEqual(row["entry_point"], "add")
        self.assertEqual(len(row["prompt"]), 2)
        self.assertIn("Complete the following Python function:", row["prompt"][1]["content"])
        self.assertIn('"""Write a function to add two numbers."""', row["prompt"][1]["content"])
        self.assertIn("def check(candidate):", row["test"])
        self.assertIn("add = candidate", row["test"])


if __name__ == "__main__":
    unittest.main()
