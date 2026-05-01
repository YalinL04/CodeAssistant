"""Build mixed SFT training data from MBPP and CodeAlpaca."""

import ast
import json
import re
import textwrap
from pathlib import Path

from src.data.dataset_loader import load_codealpaca, load_mbpp
from src.data.prompts import SYSTEM_PROMPT, format_code_prompt, format_assistant_response


def analyze_solution(code: str) -> dict:
    """Analyze a solution using AST to extract structural information."""
    analysis = {
        "has_loops": False,
        "has_recursion": False,
        "has_conditionals": False,
        "loop_types": [],
        "data_structures": [],
        "builtin_functions": [],
        "estimated_complexity": "O(n)",
        "max_loop_depth": 0,
    }

    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return analysis

    func_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_names.add(node.name)

    max_depth = [0]

    def _count_loops(node, depth=0):
        if isinstance(node, ast.For):
            analysis["has_loops"] = True
            analysis["loop_types"].append("for")
            depth += 1
            max_depth[0] = max(max_depth[0], depth)
        elif isinstance(node, ast.While):
            analysis["has_loops"] = True
            analysis["loop_types"].append("while")
            depth += 1
            max_depth[0] = max(max_depth[0], depth)
        elif isinstance(node, ast.If):
            analysis["has_conditionals"] = True
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name in func_names:
                    analysis["has_recursion"] = True
                if name in ("set", "dict", "list", "tuple", "sorted",
                            "enumerate", "zip", "map", "filter", "sum",
                            "min", "max", "len", "range", "reversed"):
                    analysis["builtin_functions"].append(name)
                if name in ("set", "dict", "defaultdict", "Counter"):
                    analysis["data_structures"].append(name)

        for child in ast.iter_child_nodes(node):
            _count_loops(child, depth)

    _count_loops(tree)
    analysis["max_loop_depth"] = max_depth[0]

    # Estimate complexity
    if analysis["has_recursion"]:
        analysis["estimated_complexity"] = "depends on recursion depth"
    elif max_depth[0] >= 3:
        analysis["estimated_complexity"] = f"O(n^{max_depth[0]})"
    elif max_depth[0] == 2:
        analysis["estimated_complexity"] = "O(n^2)"
    elif max_depth[0] == 1:
        if "sorted" in analysis["builtin_functions"]:
            analysis["estimated_complexity"] = "O(n log n)"
        else:
            analysis["estimated_complexity"] = "O(n)"
    else:
        analysis["estimated_complexity"] = "O(1)"

    return analysis


def build_reasoning_trace(problem_text: str, code: str) -> str:
    """
    Build a chain-of-thought reasoning trace from a MBPP problem.

    Args:
        problem_text: Natural language problem description
        code: The canonical solution code
    """
    analysis = analyze_solution(code)

    # Extract function name from code
    func_match = re.search(r"def\s+(\w+)\s*\(", code)
    func_name = func_match.group(1) if func_match else "solution"

    # Build reasoning components
    parts = []

    # 1. Problem understanding
    parts.append(f"Let me analyze this problem:\n")
    parts.append(f"**Task**: {problem_text.strip()}")

    # 2. Approach
    approach_parts = []
    if analysis["has_recursion"]:
        approach_parts.append("use a recursive approach")
    if analysis["has_loops"]:
        loop_desc = " and ".join(set(analysis["loop_types"]))
        approach_parts.append(f"iterate using {loop_desc} loop(s)")
    if analysis["data_structures"]:
        ds = ", ".join(set(analysis["data_structures"]))
        approach_parts.append(f"leverage {ds} for efficient lookups")
    if analysis["builtin_functions"]:
        unique_builtins = list(set(analysis["builtin_functions"]))
        builtins = ", ".join(unique_builtins[:3])
        approach_parts.append(f"use built-in functions like {builtins}")
    if analysis["has_conditionals"]:
        approach_parts.append("apply conditional checks")

    if approach_parts:
        parts.append(f"\n**Approach**: I'll {', '.join(approach_parts)}.")

    # 3. Complexity
    parts.append(f"\n**Complexity**: Time {analysis['estimated_complexity']}")

    # 4. Implementation note
    parts.append(f"\nLet me implement the `{func_name}` function.")

    return "\n".join(parts)


def build_sft_sample(problem_text: str, code: str, test_list: list[str]) -> dict:
    """
    Build a single SFT training sample.

    Returns a dict with 'messages' key in conversation format.
    """
    # Build the user prompt from problem description
    # Extract function signature from the code
    func_match = re.search(r"(def\s+\w+\s*\([^)]*\)\s*(?:->.*?)?:)", code)
    if func_match:
        sig = func_match.group(1)
        # Build a prompt with signature and docstring
        docstring = f'    """{problem_text.strip()}"""'
        user_content = format_code_prompt(f"{sig}\n{docstring}\n")
    else:
        user_content = f"Write a Python function to solve the following problem:\n\n{problem_text.strip()}"

    # Build reasoning trace
    thinking = build_reasoning_trace(problem_text, code)
    
    # Format assistant response
    # assistant_content = format_assistant_response(thinking, code)
    assistant_content = code
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def build_codealpaca_prompt(instruction: str, input_text: str = "") -> str:
    """Build a user prompt from CodeAlpaca instruction and optional input."""
    prompt = instruction.strip()
    if input_text and input_text.strip():
        prompt = f"{prompt}\n\nExample Input:\n{input_text.strip()}"
    return prompt


def _looks_like_python_code(problem: str, code: str) -> bool:
    """Heuristic to decide whether a response is mostly Python code."""
    if "Python" in problem or "python" in problem:
        return True
    foreign = ["C", "C++", "Java", "SQL", "JavaScript", "HTML", "Bash", "bash", "C#"]
    for lang in foreign:
        if lang in problem:
            return False
    if "#include" in code:
        return False
    stripped = code.strip()
    if not stripped:
        return False
    if "```python" in stripped:
        return True

    code_markers = (
        "def ",
        "class ",
        "import ",
        "from ",
        "if __name__ ==",
        "return ",
        "for ",
        "while ",
        "try:",
        "@",
    )
    return any(marker in stripped for marker in code_markers)


def build_codealpaca_sample(
    instruction: str,
    output: str,
    input_text: str = "",
) -> dict:
    """Build a single chat-style SFT sample from CodeAlpaca."""
    user_content = build_codealpaca_prompt(instruction, input_text)
    assistant_content = output.strip()

    if _looks_like_python_code(instruction, assistant_content):
        thinking = (
            "I will follow the instruction, choose a direct implementation, "
            "and return only the requested Python solution."
        )
        # assistant_content = format_assistant_response(thinking, assistant_content)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def build_sft_dataset(
    output_dir: str = "data/processed",
    codealpaca_val_size: float = 0.05,
    mbpp_repeat: int = 5,
):
    """
    Build mixed SFT training and validation datasets from MBPP and CodeAlpaca.

    Writes sft_train.jsonl and sft_val.jsonl to output_dir.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load MBPP splits
    train_ds = load_mbpp("train")
    val_ds = load_mbpp("validation")
    codealpaca_ds = load_codealpaca("train")
    codealpaca_split = codealpaca_ds.train_test_split(
        test_size=codealpaca_val_size,
        seed=42,
        shuffle=True,
    )

    # Build training samples
    train_samples = []
    for _ in range(max(1, mbpp_repeat)):
        for sample in train_ds:
            try:
                sft_sample = build_sft_sample(
                    problem_text=sample["text"],
                    code=sample["code"],
                    test_list=sample["test_list"],
                )
                train_samples.append(sft_sample)
            except Exception as e:
                print(f"Warning: skipping MBPP train task {sample.get('task_id', '?')}: {e}")

    for sample in codealpaca_split["train"]:
        try:
            if _looks_like_python_code(sample["instruction"], sample["output"]):
                train_samples.append(
                    build_codealpaca_sample(
                        instruction=sample["instruction"],
                        input_text=sample.get("input", ""),
                        output=sample["output"],
                    )
                )
        except Exception as e:
            print(f"Warning: skipping CodeAlpaca train sample: {e}")

    # Build validation samples
    val_samples = []
    for sample in val_ds:
        try:
            sft_sample = build_sft_sample(
                problem_text=sample["text"],
                code=sample["code"],
                test_list=sample["test_list"],
            )
            val_samples.append(sft_sample)
        except Exception as e:
            print(f"Warning: skipping MBPP val task {sample.get('task_id', '?')}: {e}")

    for sample in codealpaca_split["test"]:
        try:
            if _looks_like_python_code(sample["instruction"], sample["output"]):
                val_samples.append(
                    build_codealpaca_sample(
                        instruction=sample["instruction"],
                        input_text=sample.get("input", ""),
                        output=sample["output"],
                    )
                )
        except Exception as e:
            print(f"Warning: skipping CodeAlpaca val sample: {e}")

    # Write to JSONL
    train_file = output_path / "sft_train.jsonl"
    val_file = output_path / "sft_val.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(
        f"Built {len(train_samples)} train samples "
        f"(MBPP x{max(1, mbpp_repeat)} + CodeAlpaca) -> {train_file}"
    )
    print(f"Built {len(val_samples)} val samples (MBPP + CodeAlpaca) -> {val_file}")
    return str(train_file), str(val_file)


if __name__ == "__main__":
    build_sft_dataset()
