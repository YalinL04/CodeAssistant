"""Tests for SFT dataset building helpers."""

import os
import sys
import unittest
from unittest.mock import patch

from datasets import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.sft_data_builder import (
    build_codealpaca_prompt,
    build_codealpaca_sample,
    build_sft_dataset,
)


class TestSftDataBuilder(unittest.TestCase):
    def test_build_codealpaca_prompt_with_input(self):
        prompt = build_codealpaca_prompt(
            instruction="Write a Python function.",
            input_text="The function should return the square of n.",
        )
        self.assertIn("Write a Python function.", prompt)
        self.assertIn("Additional context:", prompt)
        self.assertIn("square of n", prompt)

    def test_build_codealpaca_sample_wraps_python_code(self):
        row = build_codealpaca_sample(
            instruction="Implement add.",
            output="def add(a, b):\n    return a + b\n",
        )
        self.assertEqual(row["messages"][0]["role"], "system")
        self.assertEqual(row["messages"][1]["role"], "user")
        self.assertEqual(row["messages"][2]["role"], "assistant")
        self.assertIn("<think>", row["messages"][2]["content"])
        self.assertIn("```python", row["messages"][2]["content"])

    @patch("src.data.sft_data_builder.load_codealpaca")
    @patch("src.data.sft_data_builder.load_mbpp")
    def test_build_sft_dataset_mixes_mbpp_and_codealpaca(
        self,
        mock_load_mbpp,
        mock_load_codealpaca,
    ):
        mbpp_train = Dataset.from_list(
            [
                {
                    "task_id": 1,
                    "text": "Return the sum of two numbers.",
                    "code": "def add(a, b):\n    return a + b\n",
                    "test_list": ["assert add(1, 2) == 3"],
                }
            ]
        )
        mbpp_val = Dataset.from_list(
            [
                {
                    "task_id": 2,
                    "text": "Return whether n is even.",
                    "code": "def is_even(n):\n    return n % 2 == 0\n",
                    "test_list": ["assert is_even(2) is True"],
                }
            ]
        )
        codealpaca = Dataset.from_list(
            [
                {
                    "instruction": "Write a hello-world function.",
                    "input": "",
                    "output": "def hello():\n    return 'hello'\n",
                },
                {
                    "instruction": "Explain Python lists.",
                    "input": "",
                    "output": "Python lists are mutable sequences.",
                },
            ]
        )

        def mbpp_side_effect(split):
            return mbpp_train if split == "train" else mbpp_val

        mock_load_mbpp.side_effect = mbpp_side_effect
        mock_load_codealpaca.return_value = codealpaca

        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tmp_test_sft_data",
        )
        try:
            train_file, val_file = build_sft_dataset(
                output_dir=output_dir,
                codealpaca_val_size=0.5,
                mbpp_repeat=2,
            )
            self.assertTrue(os.path.exists(train_file))
            self.assertTrue(os.path.exists(val_file))

            with open(train_file, "r", encoding="utf-8") as f:
                train_lines = f.readlines()
            with open(val_file, "r", encoding="utf-8") as f:
                val_lines = f.readlines()

            self.assertEqual(len(train_lines), 3)
            self.assertEqual(len(val_lines), 2)
        finally:
            if os.path.exists(output_dir):
                for name in os.listdir(output_dir):
                    os.remove(os.path.join(output_dir, name))
                os.rmdir(output_dir)


if __name__ == "__main__":
    unittest.main()
