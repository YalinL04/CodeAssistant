"""Recover a merged model from a saved PEFT adapter without retraining."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_adapter(
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    torch_dtype: str = "bfloat16",
):
    """
    Merge a PEFT adapter into a base model and save a standalone model.

    Typical recovery flow for this repo:
    - base_model_path: outputs/sft/merged
    - adapter_path: outputs/grpo/final
    - output_dir: outputs/grpo/merged_recovered
    """
    dtype = getattr(torch, torch_dtype)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=True,
    )

    print(f"Loading adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter weights into base model")
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
    print("Merge complete")


def main():
    parser = argparse.ArgumentParser(description="Merge a saved GRPO adapter")
    parser.add_argument(
        "--base_model_path",
        default="outputs/sft/merged_recovered",
        help="Base model to merge into. For this repo, use outputs/sft/merged.",
    )
    parser.add_argument(
        "--adapter_path",
        default="outputs/grpo/final",
        help="Path to the trained adapter directory.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/grpo/merged_recovered",
        help="Where to write the standalone merged model.",
    )
    parser.add_argument(
        "--torch_dtype",
        default="bfloat16",
        choices=["float16", "float32", "bfloat16"],
        help="Torch dtype used while loading the model for merging.",
    )
    args = parser.parse_args()

    merge_adapter(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        torch_dtype=args.torch_dtype,
    )


if __name__ == "__main__":
    main()
