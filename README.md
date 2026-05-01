# Agentic RL Code Generation Assistant

An end-to-end code-generation training pipeline built around **DeepSeek-Coder-6.7B-Base**, with:

- supervised fine-tuning (SFT) for chat-style code completion,
- GRPO-based reinforcement learning for reward-guided policy improvement,
- automatic benchmarking on **HumanEval** and **HumanEval+** via `evalplus`.

This repository focuses on practical experimentation: data building, reward design, LoRA-based training, adapter merging, and evaluation are all included in one codebase.

## Overview

The default workflow is:

```text
External datasets
  -> build SFT dataset
  -> train SFT adapter
  -> build GRPO prompt dataset
  -> train GRPO adapter
  -> merge/export model
  -> evaluate on HumanEval / HumanEval+
```

At a high level:

- **SFT** teaches the model the repository's chat/completion format.
- **GRPO** optimizes generations with program-level rewards.
- **Evaluation** measures pass@1 on HumanEval-style coding tasks.

## Features

- LoRA / QLoRA-ready training utilities for large causal language models
- Mixed SFT data pipeline from **MBPP** and **CodeAlpaca**
- GRPO training with custom rewards for:
  - correctness
  - readability
  - efficiency
- Sandboxed execution for code-based reward computation
- Reward sanity-check script before RL training
- Adapter merge utilities for exporting standalone merged checkpoints
- HumanEval / HumanEval+ benchmarking through `evalplus`

## Repository Layout

```text
configs/
  sft_config.yaml         Default SFT settings
  grpo_config.yaml        Default GRPO settings
  eval_config.yaml        Evaluation defaults

scripts/
  build_sft_dataset.py    Build the mixed SFT dataset
  run_sft.py              Run supervised fine-tuning
  run_grpo.py             Run GRPO training
  run_eval.py             Evaluate a trained model
  run_pipeline.sh         End-to-end pipeline script
  check_grpo_rewards.py   Pre-training reward discrimination check
  merge_sft_adapter.py    Merge SFT adapter into a standalone checkpoint
  merge_grpo_adapter.py   Merge GRPO adapter into a standalone checkpoint

src/
  data/                   Dataset loading and dataset builders
  models/                 Model, tokenizer, and LoRA utilities
  training/               SFT and GRPO trainers
  rewards/                Reward functions and sandbox execution
  evaluation/             Sample generation and evalplus integration
  utils/                  Code parsing and config helpers

tests/
  Unit tests for rewards, sandboxing, and data builders
```

## Training Pipeline

### 1. Supervised Fine-Tuning

The SFT stage builds chat-style examples from:

- **MBPP**: task-focused Python programming examples
- **CodeAlpaca-20k**: instruction-tuning data filtered toward Python-like outputs

The resulting dataset is saved as JSONL with `messages` in chat format.

Default behavior:

- base model: `deepseek-ai/deepseek-coder-6.7b-base`
- LoRA adapters on attention and MLP projection layers
- `trl.SFTTrainer`
- step-based evaluation
- early stopping on validation loss

### 2. GRPO Reinforcement Learning

GRPO trains a fresh LoRA adapter on top of the SFT policy.

Each prompt is sampled multiple times, and the generated candidates are scored by program-level rewards:

- **Correctness**: executes the candidate against a test harness in a sandbox
- **Readability**: static heuristics on naming, nesting, documentation, and line length
- **Efficiency**: static complexity heuristics plus runtime comparison against a canonical baseline, gated by full correctness

The default GRPO setup uses:

- `trl.GRPOTrainer`
- multiple generations per prompt
- weighted multi-objective rewards
- sandboxed execution for code rewards

### 3. Evaluation

Evaluation is split into two steps:

1. Generate code samples from the trained model
2. Run `evalplus` on HumanEval / HumanEval+

The repository includes code to automatically:

- detect base vs. fine-tuned model usage,
- apply the correct prompting style,
- export predictions in `evalplus`-compatible JSONL format.

## Datasets

The code currently integrates:

- **MBPP**
- **CodeAlpaca-20k**
- **HumanEval / HumanEval+**

Data loading happens dynamically through Hugging Face datasets and `evalplus`, so internet access is typically required the first time the datasets or models are fetched.

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended
- Enough VRAM for DeepSeek-Coder-6.7B training/inference

Core dependencies are listed in [requirements.txt](/inspire/hdd/project/embodied-multimodality/chenxie-25019/yalin/code/requirements.txt):

- `torch`
- `transformers`
- `trl`
- `peft`
- `bitsandbytes`
- `datasets`
- `accelerate`
- `evalplus`
- `wandb`

## Installation

```bash
conda create -n coder python=3.10
conda activate coder

pip install -r requirements.txt
```

## Quick Start

### Full Pipeline

```bash
bash scripts/run_pipeline.sh
```

### Step-by-Step

Build the SFT dataset:

```bash
python scripts/build_sft_dataset.py
```

Build the GRPO dataset:

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.data.grpo_data_builder import build_grpo_dataset
build_grpo_dataset('data/processed')
"
```

Run SFT:

```bash
python scripts/run_sft.py --config configs/sft_config.yaml
```

Run GRPO:

```bash
python scripts/run_grpo.py --config configs/grpo_config.yaml
```

(Optional) Build the GRPO prompt dataset and train:

```bash
python scripts/run_grpo.py --config configs/grpo_config.yaml --build_dataset --dataset humaneval --split train
```

Evaluate the trained model:

```bash
python scripts/run_eval.py --model_path outputs/grpo/merged
```
## Utils
### Reward Sanity Check Before GRPO

Before launching RL training, you can check whether your current reward setup has enough discrimination across sampled candidates:

```bash
python scripts/check_grpo_rewards.py --config configs/grpo_config.yaml --num_prompts 8 --num_generations 8
```

This script:

- samples multiple completions per prompt,
- computes correctness / readability / efficiency / total reward,
- reports within-prompt spread statistics such as standard deviation, range, and discriminative prompt ratio.

It is useful for catching flat or degenerate reward setups before expensive GRPO runs.

### Model Export and Adapter Merging

The repository stores adapter checkpoints during training and also supports standalone merged exports.

Useful utilities:

Merge a SFT adapter into a standalone model:

```bash
python scripts/merge_sft_adapter.py \
  --adapter_path outputs/sft/final \
  --output_dir outputs/sft/merged
```

Merge a GRPO adapter into a standalone model:

```bash
python scripts/merge_grpo_adapter.py \
  --base_model_path outputs/sft/merged \
  --adapter_path outputs/grpo/final \
  --output_dir outputs/grpo/merged
```

## Configuration

Default configs live in:

- [configs/sft_config.yaml](/inspire/hdd/project/embodied-multimodality/chenxie-25019/yalin/code/configs/sft_config.yaml)
- [configs/grpo_config.yaml](/inspire/hdd/project/embodied-multimodality/chenxie-25019/yalin/code/configs/grpo_config.yaml)
- [configs/eval_config.yaml](/inspire/hdd/project/embodied-multimodality/chenxie-25019/yalin/code/configs/eval_config.yaml)

Important knobs include:

- base model path
- LoRA rank / alpha / dropout
- sequence length
- optimizer and scheduler
- number of GRPO generations
- reward weights
- sandbox timeout and memory limit

## Prompting Format

The project uses a custom chat template for code generation. Prompts are structured as:

```text
<|begin_of_sentence|>You are an expert Python programmer...

### Instruction:
<coding task>

### Response:
<model output>
```

The codebase contains utilities for:

- extracting code blocks,
- rebuilding executable solutions from generations,
- preserving tokenizer whitespace behavior for DeepSeek-Coder.

## Testing

Run the unit tests with:

```bash
python -m pytest tests/ -v
```

Current test coverage focuses on:

- reward functions
- sandboxed execution
- data builders
- code utility helpers

## Experiment Results
### SFT
| Train Accuracy | Train Loss |
|-------|-------|
| ![img1](./img/train_acc.svg) | ![img2](./img/train_loss.svg) 

| Eval Accuracy | Eval Loss |
|-------|-------|
| ![img3](./img/eval_acc.svg) | ![img4](./img/eval_loss.svg) 

### GRPO
| Train Reward |
|-------|
| ![img1](./img/train_reward.svg) |

**Key Feature**

The generated code includes more inline comments and more details in the docstrings to improve readability.
```python
from typing import List


def below_zero(operations: List[int]) -> bool:
    """ You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
    at that point function should return True. Otherwise it should return False.
    
    Args:
        operations: A list of deposit and withdrawal operations on a bank account.
    
    Returns:
        True if the balance of the account falls below zero at any point, False otherwise.
    
    Raises:
        TypeError: If the input is not a list.
    
    Example:
        >>> below_zero([1, 2, 3])
        False
        >>> below_zero([1, 2, -4, 5])
        True
    """
    
    # Check if the input is a list
    if not isinstance(operations, list):
        raise TypeError("Input must be a list")
    
    # If the list is empty, return False
    if not operations:
        return False
    
    # Initialize the balance to zero
    balance = 0
    
    # Iterate over the operations
    for operation in operations:
        # Update the balance based on the operation
        balance += operation
        
        # If the balance falls below zero, return True
        if balance < 0:
            return True
    
    # If the balance never falls below zero, return False
    return False
```

### Evaluation
|  | Humaneval | Humaneval+ |
|-------|-------|---------|
| base |  0.457 | 0.384 |
| SFT |  0.616  | 0.530 |
| GRPO |  0.732  | 0.634 |

Experiments are conducted on 4 NVIDIA 4090.
## Notes

- The repository is designed for experimentation, so some defaults may evolve as reward design and training strategy change.
- If you modify reward functions, it is strongly recommended to run the GRPO reward sanity-check script before training.
