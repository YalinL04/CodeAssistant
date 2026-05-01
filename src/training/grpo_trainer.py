"""GRPO training pipeline using trl.GRPOTrainer."""

import json
from pathlib import Path

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from src.models.model_loader import (
    load_model_for_training,
    load_tokenizer,
    get_lora_config,
)
from src.rewards.correctness import correctness_reward
from src.rewards.readability import readability_reward
from src.rewards.efficiency import efficiency_reward
from src.utils.config_loader import load_config


def load_grpo_dataset(file_path: str) -> Dataset:
    """Load GRPO prompt dataset from JSONL file."""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return Dataset.from_list(samples)


def run_grpo_training(config_path: str = "configs/grpo_config.yaml"):
    """
    Run the full GRPO training pipeline.

    Steps:
    1. Load config
    2. Load SFT checkpoint with fresh QLoRA adapter
    3. Load GRPO prompt dataset
    4. Configure reward functions
    5. Train with GRPOTrainer
    6. Save final model
    """
    # Load configuration
    config = load_config(config_path)
    model_cfg = config["model"]
    training_cfg = config["training"]
    grpo_cfg = config["grpo"]
    data_cfg = config["data"]

    print(f"Loading model: {model_cfg['name']}")

    # Load model from SFT checkpoint
    model = load_model_for_training(config)
    tokenizer = load_tokenizer(model_cfg["name"])

    # Load GRPO prompt dataset
    print(f"Loading GRPO prompts: {data_cfg['prompt_file']}")
    dataset = load_grpo_dataset(data_cfg["prompt_file"])
    print(f"GRPO prompts: {len(dataset)}")

    # LoRA config for GRPO (fresh adapter on top of SFT model)
    lora_config = get_lora_config(config)

    # Reward functions and weights
    reward_funcs = [correctness_reward, readability_reward, efficiency_reward]
    reward_weights = grpo_cfg.get("reward_weights", [0.70, 0.15, 0.15])

    # GRPO training config
    output_dir = training_cfg["output_dir"]
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        max_steps=training_cfg.get("max_steps", 500),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=training_cfg.get("learning_rate", 5e-5),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.05),
        max_grad_norm=training_cfg.get("max_grad_norm", 0.5),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        bf16=training_cfg.get("bf16", True),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
        logging_steps=training_cfg.get("logging_steps", 5),
        save_steps=training_cfg.get("save_steps", 100),
        save_total_limit=training_cfg.get("save_total_limit", 5),
        seed=training_cfg.get("seed", 42),
        optim=training_cfg.get("optim", "paged_adamw_8bit"),
        report_to="wandb",
        run_name="grpo-deepseek-coder",
        # GRPO specific
        num_generations=grpo_cfg.get("num_generations", 8),
        max_completion_length=grpo_cfg.get("max_completion_length", 1024),
        temperature=grpo_cfg.get("temperature", 0.7),
        beta=grpo_cfg.get("beta", 0.001),
        loss_type=grpo_cfg.get("loss_type", "grpo"),
        scale_rewards=grpo_cfg.get("scale_rewards", True),
        reward_weights=reward_weights,
    )

    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_funcs,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Train
    print("Starting GRPO training...")
    print(f"  Reward weights: correctness={reward_weights[0]}, "
          f"readability={reward_weights[1]}, efficiency={reward_weights[2]}")
    print(f"  Generations per prompt: {grpo_cfg.get('num_generations', 8)}")
    print(f"  Max steps: {training_cfg.get('max_steps', 500)}")

    trainer.train()

    # Save final model
    final_path = Path(output_dir) / "final"
    print(f"Saving final model to {final_path}")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # Merge LoRA weights
    merged_path = Path(output_dir) / "merged"
    print(f"Merging LoRA weights and saving to {merged_path}")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))

    print("GRPO training complete!")
    return str(final_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run GRPO training")
    parser.add_argument("--config", default="configs/grpo_config.yaml")
    args = parser.parse_args()
    run_grpo_training(args.config)
