"""Utilities for extracting and processing code from model completions."""

import re
import ast
import textwrap


# Deepseek-coder special tokens and BPE artifacts to clean up
_SPECIAL_TOKEN_PATTERNS = [
    r"<jupyter_code>",
    r"<jupyter_output>",
    r"<jupyter_text>",
    r"<empty_output>",
    r"<\|end of sentence\|>",
    r"<｜end of sentence｜>",
    r"<\|begin of sentence\|>",
    r"<｜begin of sentence｜>",
    r"<\|EOT\|>",
    r"<pad>",
]


def clean_raw_output(text: str) -> str:
    """
    Clean raw model output by removing special tokens and BPE artifacts.

    Handles deepseek-coder specific tokens like <jupyter_code>,
    byte-level BPE artifacts (Ġ -> space, Ċ -> newline), etc.
    """
    # Remove special tokens
    for pattern in _SPECIAL_TOKEN_PATTERNS:
        text = re.sub(pattern, "", text)

    # Remove BPE byte-level artifacts if present
    # Ġ (U+0120) represents a leading space in GPT-2 BPE
    # Ċ (U+010A) represents a newline in GPT-2 BPE
    text = text.replace("\u0120", " ")
    text = text.replace("\u010a", "\n")

    return text


def extract_code_from_completion(completion: str) -> str:
    """
    Extract Python code from a model completion that may contain
    <think>...</think> tags and ```python...``` code blocks.
    """
    # Clean up special tokens first
    completion = clean_raw_output(completion)

    # Remove thinking block
    text = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL).strip()

    # Try to extract from fenced code block
    code_blocks = re.findall(r"```python\s*\n(.*?)```", text, flags=re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    # Try generic fenced block
    code_blocks = re.findall(r"```\s*\n(.*?)```", text, flags=re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    # Fallback: return everything after ### Response: if present
    if "### Response:" in text:
        text = text.split("### Response:")[-1].strip()

    return text.strip()


def extract_function_completion(completion: str, prompt: str) -> str:
    """
    Extract the function body from a base model's completion.

    For base/completion models that receive the raw HumanEval prompt
    and generate the function body directly (no chat format).

    The completion is truncated at the first sign of a new top-level
    definition (def/class) or excessive blank lines.
    """
    completion = clean_raw_output(completion)

    # Truncate at new top-level definitions
    # These patterns indicate the model has finished the target function
    stop_patterns = [
        r"\nclass\s",        # new class definition
        r"\n# ----",         # separator comments
        r'\nif __name__',    # main guard
        r'\nprint\(',        # top-level print (test invocation)
    ]

    # Also stop at a new function def that is NOT indented (top-level)
    # but allow nested def (indented)
    lines = completion.split("\n")
    result_lines = []
    for line in lines:
        # Stop at new top-level function definition
        if line and not line[0].isspace() and line.startswith("def "):
            break
        # Stop at known stop patterns
        current_text = "\n".join(result_lines + [line])
        should_stop = False
        for pat in stop_patterns:
            if re.search(pat, "\n" + line):
                should_stop = True
                break
        if should_stop:
            break
        result_lines.append(line)

    result = "\n".join(result_lines)

    # Remove trailing blank lines
    result = result.rstrip()

    return result


def extract_function_body(code: str, entry_point: str) -> str:
    """Extract the body of a specific function from code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            lines = code.split("\n")
            # Get lines from function body start to end
            start = node.body[0].lineno - 1
            end = node.end_lineno
            return "\n".join(lines[start:end])
    return code


def strip_prompt_prefix(generated_code: str, prompt: str) -> str:
    """Remove the prompt prefix if the model re-generated it."""
    prompt_stripped = prompt.strip()
    gen_stripped = generated_code.strip()
    if gen_stripped.startswith(prompt_stripped):
        return gen_stripped[len(prompt_stripped):]
    return generated_code


def count_loop_nesting(code: str) -> int:
    """Count maximum loop nesting depth using AST."""
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return 0

    max_depth = [0]

    def _walk(node, depth):
        if isinstance(node, (ast.For, ast.While)):
            depth += 1
            max_depth[0] = max(max_depth[0], depth)
        for child in ast.iter_child_nodes(node):
            _walk(child, depth)

    _walk(tree, 0)
    return max_depth[0]


def count_nesting_depth(code: str) -> int:
    """Count maximum indentation nesting depth."""
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return 0

    max_depth = [0]

    def _walk(node, depth):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            depth += 1
            max_depth[0] = max(max_depth[0], depth)
        for child in ast.iter_child_nodes(node):
            _walk(child, depth)

    _walk(tree, 0)
    return max_depth[0]


def get_variable_names(code: str) -> list[str]:
    """Extract all variable names from code using AST."""
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return []

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.FunctionDef):
            for arg in node.args.args:
                names.add(arg.arg)
    return list(names)


def has_docstring(code: str) -> bool:
    """Check if the code contains a docstring."""
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)):
                return True
    return False


def uses_efficient_structures(code: str) -> dict:
    """Check for usage of efficient data structures."""
    indicators = {
        "set": bool(re.search(r"\bset\s*\(", code)),
        "dict": bool(re.search(r"\bdict\s*\(", code)) or bool(re.search(r"\{\s*\w+\s*:", code)),
        "defaultdict": "defaultdict" in code,
        "Counter": "Counter" in code,
        "sorted": "sorted(" in code,
        "enumerate": "enumerate(" in code,
        "zip": "zip(" in code,
        "list_comprehension": bool(re.search(r"\[.+\bfor\b.+\bin\b", code)),
    }
    return indicators
