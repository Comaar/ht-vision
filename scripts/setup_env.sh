#!/bin/bash
# Environment setup for HT-Vision training scripts
# Usage: source scripts/setup_env.sh

# ==============================================================================
# REQUIRED: Update these paths for your environment
# ==============================================================================

# Working directory (where model weights and datasets are located)
export YOLO_WORKDIR="/path/to/your/working/directory"

# Path to dataset configuration file
export YOLO_DATA_YAML="/path/to/your/dataset/data.yaml"

# Output directory for training results
export YOLO_OUTPUT_DIR="/path/to/your/output/directory"

# ==============================================================================
# OPTUNA-SPECIFIC (for hyperparameter optimization scripts)
# ==============================================================================

# Output directory for Optuna Phase 1
export OPTUNA_OUTPUT_DIR="/path/to/optuna/output"

# Base model weights
export YOLO_BASE_MODEL="yolo11m.pt"

# Model weights for Phase 2 (can be same as BASE_MODEL or a fine-tuned version)
export YOLO_WEIGHTS="yolo11m.pt"

# ==============================================================================
# Verify configuration
# ==============================================================================
echo "HT-Vision Environment Configuration:"
echo "  YOLO_WORKDIR:     $YOLO_WORKDIR"
echo "  YOLO_DATA_YAML:   $YOLO_DATA_YAML"
echo "  YOLO_OUTPUT_DIR:  $YOLO_OUTPUT_DIR"
echo "  OPTUNA_OUTPUT_DIR: $OPTUNA_OUTPUT_DIR"
echo "  YOLO_BASE_MODEL:  $YOLO_BASE_MODEL"
echo "  YOLO_WEIGHTS:     $YOLO_WEIGHTS"
echo ""
echo "To run scripts:"
echo "  python model_comparison/yolo_5models_comparison_training_resume.py"
echo "  python model/bayesian_hp_optimization/optuna_phase1_core_hp_search.py"
echo "  python model/bayesian_hp_optimization/optuna_phase2_data_augmentation_hp_search.py"
