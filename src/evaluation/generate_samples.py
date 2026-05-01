"""Generate code completions for HumanEval evaluation."""

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


def generate_samples(
    model_path: str,
    output_file: str = "outputs/eval/samples.jsonl",
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    do_sample: bool = False,
    num_samples: int = 1,
) -> str:
    """
    Generate completions for all HumanEval problems.

    Automatically detects base vs. fine-tuned models:
    - Base model: uses raw completion (feed function signature directly)
    - Fine-tuned model: uses chat template with <think> reasoning

    Args:
        model_path: Path to the trained model (or HuggingFace model ID)
        output_file: Where to save generated samples
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for greedy)
        do_sample: Whether to use sampling
        num_samples: Number of samples per problem (1 for pass@1)

    Returns:
        Path to the output file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    print(f"Loading model from {model_path}")
    model, tokenizer = load_model_for_inference(model_path)
    model.eval()

    # Detect model type
    base_model = is_base_model(model_path)
    if base_model:
        print("Detected BASE model -> using completion-style prompting")
    else:
        print("Detected FINE-TUNED model -> using chat template")

    # Get stop token IDs (includes <|EOT|>, <jupyter_code>, etc.)
    stop_token_ids = get_stop_token_ids(tokenizer)
    print(f"Stop token IDs: {stop_token_ids}")

    # Load HumanEval problems
    problems = load_humaneval()
    print(f"Generating solutions for {len(problems)} problems...")

    samples = []

    for task_id, problem in tqdm(problems.items(), desc="Generating"):
        prompt = problem["prompt"]

        if base_model:
            # Base model: feed raw HumanEval prompt for completion
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        else:
            # Fine-tuned model: use chat template
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_code_prompt(prompt)},
            ]
            
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            # input_ids = inputs["input_ids"]
        for _ in range(num_samples):
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": stop_token_ids,
                }
                if do_sample and temperature > 0:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = 0.95

                output = model.generate(input_ids, **gen_kwargs)

            # Decode only the generated part
            generated_ids = output[0][input_ids.shape[1]:]
            completion_text = tokenizer.decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            if base_model:
                # Base model: keep a full solution so evalplus does not need to
                # reconstruct it from prompt + completion.
                completion = extract_function_completion(completion_text, prompt)
                solution = prompt + completion
                print(solution)
                samples.append({
                    "task_id": task_id,
                    "solution": solution,
                })
            else:
                # Fine-tuned model: extract from chat format
                # print(completion_text)
                code = extract_code_from_completion(completion_text)
                code = strip_prompt_prefix(code, prompt)
                print(code)
                samples.append({
                    "task_id": task_id,
                    "completion": code,
                })

    # Write samples in evalplus format
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Generated {len(samples)} samples -> {output_file}")
    return output_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_file", default="outputs/eval/samples.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()

    generate_samples(
        model_path=args.model_path,
        output_file=args.output_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
        num_samples=args.num_samples,
    )
