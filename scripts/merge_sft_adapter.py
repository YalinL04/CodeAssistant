"""Merge the SFT LoRA adapter into a non-quantized base model."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_base_model(adapter_path: str, base_model_path: str | None) -> str:
    """Resolve the base model path from args or adapter metadata."""
    if base_model_path:
        return base_model_path

    config_path = Path(adapter_path) / "adapter_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(f"Missing base_model_name_or_path in {config_path}")
    return base_model


def merge_adapter(
    adapter_path: str,
    output_dir: str,
    base_model_path: str | None = None,
    torch_dtype: str = "float16",
):
    """
    Merge an SFT adapter into a full-precision/half-precision base model.

    This intentionally avoids loading a quantized base model so the merged
    checkpoint is a clean standalone model.
    """
    base_model_path = resolve_base_model(adapter_path, base_model_path)
    dtype = getattr(torch, torch_dtype)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading non-quantized base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=True,
    )

    print(f"Loading adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter into base model")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {output_dir}")
    merged_model.save_pretrained(output_dir)

    print("Saving tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    local_template = Path(adapter_path) / "chat_template.jinja"
    if local_template.exists():
        tokenizer.chat_template = local_template.read_text(encoding="utf-8")
    tokenizer.save_pretrained(output_dir)

    print("SFT merge complete")


def main():
    parser = argparse.ArgumentParser(description="Merge the SFT LoRA adapter")
    parser.add_argument(
        "--adapter_path",
        default="outputs/sft/final",
        help="Path to the saved SFT adapter directory.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/sft/merged_recovered",
        help="Where to write the merged standalone model.",
    )
    parser.add_argument(
        "--base_model_path",
        default=None,
        help="Optional base model path. If omitted, use adapter_config.json.",
    )
    parser.add_argument(
        "--torch_dtype",
        default="bfloat16",
        choices=["float16", "float32", "bfloat16"],
        help="Torch dtype for loading the non-quantized base model.",
    )
    args = parser.parse_args()

    merge_adapter(
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        base_model_path=args.base_model_path,
        torch_dtype=args.torch_dtype,
    )


if __name__ == "__main__":
    main()
