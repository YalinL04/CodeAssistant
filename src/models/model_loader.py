"""Model and tokenizer loading with QLoRA configuration."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from src.data.prompts import CHAT_TEMPLATE

# Known base (completion-only) model identifiers
_BASE_MODEL_KEYWORDS = ["base", "foundation"]


def is_base_model(model_name: str) -> bool:
    """
    Detect if a model is a base/completion model (vs. instruct/chat).

    Base models should NOT use chat templates for inference.
    """
    name_lower = model_name.lower().split("/")[-1]
    # Check for explicit base model indicators
    for kw in _BASE_MODEL_KEYWORDS:
        if kw in name_lower and "instruct" not in name_lower:
            return True
    return False


def get_quantization_config(config: dict) -> BitsAndBytesConfig:
    """Create BitsAndBytes quantization config from YAML config."""
    quant_cfg = config.get("quantization", {})
    if not quant_cfg.get("load_in_4bit", False):
        return None

    compute_dtype = getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )


def get_lora_config(config: dict) -> LoraConfig:
    """Create LoRA config from YAML config."""
    lora_cfg = config.get("lora", {})
    task_type_str = lora_cfg.get("task_type", "CAUSAL_LM")
    task_type = getattr(TaskType, task_type_str)

    return LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        task_type=task_type,
        bias=lora_cfg.get("bias", "none"),
    )


def load_tokenizer(model_name: str, use_chat_template: bool = True) -> AutoTokenizer:
    """
    Load and configure the tokenizer.

    Args:
        model_name: HuggingFace model name or local path
        use_chat_template: If True, register custom chat template.
                           Set False for base model inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Set pad token (required for batched training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Register custom chat template only when needed
    if use_chat_template:
        tokenizer.chat_template = CHAT_TEMPLATE

    if tokenizer_has_broken_whitespace(tokenizer):
        print(
            "WARNING: tokenizer collapses spaces/newlines during encode-decode. "
            "This usually indicates an incompatible transformers/tokenizer "
            "version for DeepSeek-Coder."
        )

    return tokenizer


def tokenizer_has_broken_whitespace(tokenizer: AutoTokenizer) -> bool:
    """
    Detect tokenizer configurations that collapse whitespace/newlines.

    DeepSeek-Coder should preserve basic spaces and line breaks. If this
    round-trip fails, generation quality and output formatting become invalid.
    """
    probe = "a b\nreturn x + 1\n"
    ids = tokenizer.encode(probe, add_special_tokens=False)
    decoded = tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return (" " not in decoded) or ("\n" not in decoded)


def _resolve_tokenizer_source(model_path: str) -> str:
    """
    Resolve the tokenizer source for a local model path.

    PEFT adapter directories often save a tokenizer.json that reloads with
    broken whitespace behavior for DeepSeek-Coder. In that case we should
    reuse the base model tokenizer recorded in adapter_config.json.
    """
    path = Path(model_path)
    if not path.exists() or not path.is_dir():
        return model_path

    adapter_config = path / "adapter_config.json"
    if adapter_config.exists():
        with open(adapter_config, "r", encoding="utf-8") as f:
            config = json.load(f)
        base_model = config.get("base_model_name_or_path")
        if base_model:
            return base_model

    return model_path


def _load_local_chat_template(model_path: str) -> str | None:
    """Load a saved chat template from a local model directory if present."""
    template_path = Path(model_path) / "chat_template.jinja"
    if not template_path.exists():
        return None
    return template_path.read_text(encoding="utf-8")


def get_stop_token_ids(tokenizer: AutoTokenizer) -> list[int]:
    """
    Get a list of token IDs that should stop generation.

    Includes eos_token and deepseek-coder specific stop tokens
    like <|EOT|>, <jupyter_code>, etc.
    """
    stop_ids = [tokenizer.eos_token_id]

    # Add known deepseek-coder special tokens if they exist
    special_tokens = [
        "<|EOT|>",
        "<jupyter_code>",
        "<jupyter_output>",
        "<jupyter_text>",
    ]
    for token_str in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        # convert_tokens_to_ids returns unk_token_id if not found
        if token_id != tokenizer.unk_token_id and token_id not in stop_ids:
            stop_ids.append(token_id)

    return stop_ids


def load_model_for_training(config: dict) -> AutoModelForCausalLM:
    """
    Load the base model with quantization for training.

    Args:
        config: Full YAML config dict with 'model' and 'quantization' sections

    Returns:
        Quantized model ready for LoRA attachment
    """
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "deepseek-ai/deepseek-coder-6.7b-base")
    torch_dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
    attn_impl = model_cfg.get("attn_implementation", None)

    quant_config = get_quantization_config(config)

    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }
    if quant_config:
        kwargs["quantization_config"] = quant_config
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    # Prepare for k-bit training if quantized
    if quant_config:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    return model


def load_model_for_inference(model_path: str, torch_dtype: str = "bfloat16"):
    """
    Load a model for inference.

    Automatically detects base vs. fine-tuned models and configures
    the tokenizer accordingly (no chat template for base models).
    """
    dtype = getattr(torch, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )

    # Base models should NOT use chat template for inference
    use_chat = not is_base_model(model_path)
    tokenizer_source = _resolve_tokenizer_source(model_path)
    tokenizer = load_tokenizer(tokenizer_source, use_chat_template=use_chat)

    # Preserve any locally saved chat template even when the tokenizer is
    # sourced from the base model.
    if use_chat:
        local_template = _load_local_chat_template(model_path)
        if local_template is not None:
            tokenizer.chat_template = local_template

    return model, tokenizer
