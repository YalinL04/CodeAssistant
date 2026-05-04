import json
from pathlib import Path

import torch
from tqdm import tqdm

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

model_path = "outputs/grpo/final"
model, tokenizer = load_model_for_inference(model_path)
model.eval()

prompt = "Write a quick sort algorithm in Python."
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
        "max_new_tokens": 1024,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": stop_token_ids,
    }
    output = model.generate(input_ids, **gen_kwargs)

generated_ids = output[0][input_ids.shape[1]:]
completion_text = tokenizer.decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(completion_text)