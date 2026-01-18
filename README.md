# Hydrotwin Vision

A research-oriented repository for **training, evaluating, and comparing YOLO-based object detection models** applied to aquatic species detection in underwater and aquarium environments.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

This repository collects **datasets, training pipelines, evaluation utilities, and comparison experiments** developed for the HT-Vision project.

**Main objectives:**

- Investigate **object detection performance** across multiple YOLO architectures
- Evaluate **training strategies and dataset composition choices**
- Compare models using **consistent datasets and evaluation metrics**

The project is designed as a **modular research framework**, supporting reproducibility and structured experimentation.

---

### Dataset Preparation

Navigate to `dataset_composition/` and run the notebooks in order:

1. `01_annotation_converter.ipynb` – Convert annotations to YOLO format
2. `02_datasets_audit.ipynb` – Audit and clean datasets
3. `03_merge_dataset.ipynb` – Merge datasets and create splits

### Model Training

Use the two-stage training pipeline in `model/training/`:

```bash
jupyter notebook model/training/YOLO11_HT_Vision_two_stage_training.ipynb
```

### Model Comparison

Run the comparison script:

```bash
python model_comparison/yolo_5models_comparison_training_resume.py
```

> **Note:** Update the paths in scripts to match your local environment before running.

---

## Models and Experiments

The repository focuses on **YOLO-based object detection models**, including:

- YOLOv8 (multiple scales)
- YOLO11 (nano, small, medium variants)
- Custom two-stage training strategies
- Dataset-specific fine-tuned models

The experiments address:
- single-stage vs two-stage training strategies,
- cross-model performance comparison,
- qualitative inference analysis on unseen images.

---

## Repository Structure

```text
ht-vision/
├── dataset_composition/       # Dataset preparation and merging
│   ├── 01_annotation_converter.ipynb
│   ├── 02_datasets_audit.ipynb
│   ├── 03_merge_dataset.ipynb
│   └── README.md
│
├── model/                     # Training pipelines and optimization
│   ├── training/
│   │   └── YOLO11_HT_Vision_two_stage_training.ipynb
│   ├── results/
│   │   ├── training_curves.ipynb
│   │   └── stage1_vs_stage3_test_metrics.csv
│   ├── bayesian_hp_optimization/
│   │   ├── optuna_phase1_core_hp_search.py
│   │   ├── optuna_phase2_data_augmentation_hp_search.py
│   │   └── README.md
│   └── README.md
│
├── model_comparison/          # Multi-model comparison experiments
│   ├── ds_aquarium_combined.ipynb
│   ├── yolo_5models_comparison_training_resume.py
│   ├── evaluation_results.csv
│   ├── configs/
│   ├── inference_images/
│   └── README.md
│
├── cross_domain_analysis/     # Cross-domain generalization studies
│   ├── 00_prepare_cross_domain_scenarios.ipynb
│   ├── 01_distortion_analysis.ipynb
│   ├── 02_distortion_analysis.ipynb
│   ├── 03_yolo11m_cross_domain_training.ipynb
│   ├── 04_distortion_correlation.ipynb
│   └── README.md
|
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Dataset Composition

The `dataset_composition/` module addresses:
- annotation format conversion,
- dataset auditing and quality assessment,
- controlled dataset merging strategies.

These steps ensure **dataset consistency across experiments** and enable reliable model comparison.

---

## Model Training

The `model/` module includes:
- standard and two-stage training pipelines,
- Bayesian hyperparameter optimization using Optuna,
- structured experiment tracking and result analysis.

Training strategies are designed to evaluate:
- convergence behavior,
- robustness to dataset variability,
- performance scalability across model sizes.

---

## Evaluation and Comparison

The `model_comparison/` module provides:

- Unified evaluation datasets
- Cross-model metric comparison
- Qualitative inference visualizations

Evaluation focuses on:

- Precision, recall, and mAP metrics
- Stage-wise performance differences
- Generalization across different aquatic environments

📖 See [model_comparison/README.md](model_comparison/README.md) for detailed results.

---

## Cross-Domain Analysis

The `cross_domain_analysis/` module investigates:

- Domain shift effects on model performance
- Impact of image distortions across datasets
- Cross-domain generalization strategies

📖 See [cross_domain_analysis/README.md](cross_domain_analysis/README.md) for details.

---

## Research Focus

Key research dimensions explored in this repository:

- Impact of **dataset composition** on detection performance
- Trade-offs between **model size, accuracy, and efficiency**
- Effectiveness of **multi-stage training strategies**
- Model **generalization across heterogeneous visual domains**

---


## License

This project is released under the **Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)** license.

Commercial use is **not permitted** without explicit permission from the author.

See [LICENSE](LICENSE) for full terms.
