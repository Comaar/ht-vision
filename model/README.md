# Model Training

This folder contains the **two-stage training pipeline** for YOLO11 on the HT-Vision dataset, along with hyperparameter optimization and results analysis.

---

## Contents

```text
model/
├── training/
│   └── YOLO11_HT_Vision_two_stage_training.ipynb
├── results/
│   ├── training_curves.ipynb
│   └── stage1_vs_stage3_test_metrics.csv
├── bayesian_hp_optimization/
│   ├── optuna_phase1_core_hp_search.py
│   ├── optuna_phase2_data_augmentation_hp_search.py
│   └── README.md
└── README.md
```

---

## Two-Stage Training Pipeline

The notebook `training/YOLO11_HT_Vision_two_stage_training.ipynb` implements a **two-stage training pipeline** based on progressive resizing and robust resume/skip logic.

### Stage Overview

| Stage | Resolution | Description |
|-------|------------|-------------|
| Stage 1 | 640×640 | Initial training from base YOLO11 weights |
| Stage 3* | 1024×1024 | Fine-tuning from Stage 1 best checkpoint |

> **Note on naming:** Stage 3 naming is preserved for consistency with existing training artifacts.

### Output Structure

```text
Training_Results/
├── Stage1/          # 640×640 training outputs
│   └── weights/
│       ├── best.pt
│       └── last.pt
└── Stage3/          # 1024×1024 fine-tuning outputs
    └── weights/
        ├── best.pt
        └── last.pt
```

---

## Hyperparameter Optimization

See [bayesian_hp_optimization/README.md](bayesian_hp_optimization/README.md) for details on the Optuna-based optimization process.

---

## Dataset Configuration

- **Dataset source:** Unified Custom Dataset
- **Task:** Single-class object detection (`fish`)
- **Paths:** Must be adapted for your environment

