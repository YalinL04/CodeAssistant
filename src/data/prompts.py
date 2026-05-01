"""Prompt templates for the code generation assistant."""

SYSTEM_PROMPT = (
    "You are an expert Python programmer. When given a coding task, "
    "write clean, correct Python code."
)

CHAT_TEMPLATE = """\
{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}\
{%- for message in messages %}
{%- if message['role'] == 'system' %}
{{ bos_token }}{{ message['content'] }}
{%- elif message['role'] == 'user' %}

### Instruction:
{{ message['content'] }}
{%- elif message['role'] == 'assistant' %}

### Response:
{{ message['content'] }}{{ eos_token }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}

### Response:
{% endif %}\
"""


def format_code_prompt(function_signature: str) -> str:
    """Format a function signature into a user prompt."""
    return f"Complete the following Python function:\n\n```python\n{function_signature.strip()}\n```"


def format_assistant_response(thinking: str, code: str) -> str:
    """Format thinking + code into an assistant response."""
    return f"<think>\n{thinking.strip()}\n</think>\n\n```python\n{code.strip()}\n```"
