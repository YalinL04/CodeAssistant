"""SFT training pipeline using trl.SFTTrainer."""

import json
from pathlib import Path

from datasets import Dataset
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from src.models.model_loader import (
    load_model_for_training,
    load_tokenizer,
    get_lora_config,
)
from src.utils.config_loader import load_config


def load_sft_dataset(file_path: str) -> Dataset:
    """Load SFT dataset from JSONL file."""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    dataset = Dataset.from_list(samples)
    return dataset.shuffle(seed=42)

def run_sft_training(config_path: str = "configs/sft_config.yaml"):
    """
    Run the full SFT training pipeline.

    Steps:
    1. Load config
    2. Load model with QLoRA
    3. Load tokenizer with chat template
    4. Load SFT datasets
    5. Train with SFTTrainer
    6. Save final model with merged LoRA weights
    """
    # Load configuration
    config = load_config(config_path)
    model_cfg = config["model"]
    training_cfg = config["training"]
    data_cfg = config["data"]

    print(f"Loading model: {model_cfg['name']}")

    # Load model and tokenizer
    model = load_model_for_training(config)
    tokenizer = load_tokenizer(model_cfg["name"])

    # Load datasets
    print(f"Loading training data: {data_cfg['train_file']}")
    train_dataset = load_sft_dataset(data_cfg["train_file"])
    eval_dataset = load_sft_dataset(data_cfg["val_file"])

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(eval_dataset)}")

    # LoRA config
    lora_config = get_lora_config(config)

    # SFT training config
    output_dir = training_cfg["output_dir"]
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=training_cfg.get("num_train_epochs", 5),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=training_cfg.get(
            "per_device_eval_batch_size",
            training_cfg.get("per_device_train_batch_size", 2),
        ),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=training_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.1),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        bf16=training_cfg.get("bf16", True),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
        max_length=training_cfg.get("max_seq_length", 2048),
        logging_steps=training_cfg.get("logging_steps", 10),
        save_strategy="steps",
        save_steps=training_cfg.get("save_steps", 50),
        eval_strategy="steps",
        eval_steps=training_cfg.get("eval_steps", 50),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        seed=training_cfg.get("seed", 42),
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 4),
        optim=training_cfg.get("optim", "paged_adamw_8bit"),
        report_to=training_cfg.get("report_to", "wandb"),
        run_name=training_cfg.get("run_name", "sft-deepseek-coder"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_cfg.get("early_stopping_patience", 4),
                early_stopping_threshold=training_cfg.get(
                    "early_stopping_threshold", 5e-4
                ),
            )
        ],
    )

    # Train
    print("Starting SFT training...")
    trainer.train()

    # Save final model
    final_path = Path(output_dir) / "final"
    print(f"Saving final model to {final_path}")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # Optionally merge LoRA weights for easier inference
    merged_path = Path(output_dir) / "merged"
    print(f"Merging LoRA weights and saving to {merged_path}")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))

    print("SFT training complete!")
    return str(final_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SFT training")
    parser.add_argument("--config", default="configs/sft_config.yaml")
    args = parser.parse_args()
    run_sft_training(args.config)
