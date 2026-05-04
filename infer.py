import json
from pathlib import Path

import torch
from tqdm import tqdm
import argparse
from src.data.dataset_loader import load_humaneval, load_mbpp
from src.data.prompts import SYSTEM_PROMPT, format_code_prompt
from src.models.model_loader import (
    load_model_for_inference,
    get_stop_token_ids,
    is_base_model,
)
from src.utils.code_utils import (
    extract_code_from_completion,
    extract_function_completion,
    strip_prompt_prefix,
)
def main():
    parser = argparse.ArgumentParser(description="Inference on trained model.")
    parser.add_argument(
        "--model",
        default="outputs/grpo/final",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--prompt",
        default="Write a quick sort algorithm in Python.",
        help="Prompt for inference",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate",
    )
    args = parser.parse_args()
    model_path = args.model
    model, tokenizer = load_model_for_inference(model_path)
    model.eval()

    prompt = args.prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    stop_token_ids = get_stop_token_ids(tokenizer)

    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": stop_token_ids,
            "temperature": 0.7
        }
        output = model.generate(input_ids, **gen_kwargs)

    generated_ids = output[0][input_ids.shape[1]:]
    completion_text = tokenizer.decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(completion_text)
    
if __name__ == "__main__":
    main()