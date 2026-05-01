#!/bin/bash
# Full training pipeline: Data Build -> SFT -> GRPO -> Evaluation
set -e

echo "============================================================"
echo "  Agentic RL Code Generation Assistant - Full Pipeline"
echo "============================================================"

# Step 1: Build SFT dataset from MBPP & CodeAlpaca
echo ""
echo "[Step 1/5] Building SFT dataset from MBPP & CodeAlpaca..."
python scripts/build_sft_dataset.py --output_dir data/processed

# Step 2: Build GRPO dataset from HumanEval
echo ""
echo "[Step 2/5] Building GRPO dataset from HumanEval..."
python -c "
import sys; sys.path.insert(0, '.')
from src.data.grpo_data_builder import build_grpo_dataset
build_grpo_dataset('data/processed')
"

# Step 3: SFT training
echo ""
echo "[Step 3/5] Running SFT training..."
python scripts/run_sft.py --config configs/sft_config.yaml

# Step 4: GRPO training
echo ""
echo "[Step 4/5] Running GRPO training..."
python scripts/run_grpo.py --config configs/grpo_config.yaml

# Step 5: Evaluation
echo ""
echo "[Step 5/5] Running evaluation..."
python scripts/run_eval.py --model_path outputs/grpo/merged --output_dir outputs/eval

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "============================================================"
