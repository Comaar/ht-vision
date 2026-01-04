# Hydrotwin Vision

A research-oriented repository for **training, evaluating, and comparing multiple YOLO-based object detection models** applied to aquatic species detection in underwater and aquarium environments.

---

## Overview

This repository collects **datasets, training pipelines, evaluation utilities, and comparison experiments** developed for the HT-Vision project.

The main objectives are to:
- investigate **object detection performance** across multiple YOLO architectures,
- evaluate **training strategies and dataset composition choices**,
- compare models using **consistent datasets and evaluation metrics**.

The project is designed as a **modular research framework**, supporting reproducibility and structured experimentation.

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
├── dataset_composition/
│   ├── 01_annotation_converter.ipynb
│   ├── 02_datasets_audit.ipynb
│   ├── 03_merge_dataset.ipynb
│   └── README.md
│
├── model/
│   ├── training/
│   │   └── YOLO11_HT_Vision_two_stage_training.ipynb
│   ├── results/
│   │   ├── Training_curves.ipynb
│   │   └── stage1_vs_stage3_test_metrics.csv
│   ├── bayesian_hp_optimization/
│   │   ├── optuna_phase1_core_hp_search.py
│   │   └── optuna_phase2_data_augementation_hp_search.py
│   └── README.md
│
├── model_comparison/
│   ├── ds_aquarium_cobined.ipynb
│   ├── yolo_5models_comparison_training_resume.py
│   ├── evaluation_results.csv
│   ├── inference_images/
│   └── README.md
│
├── README.md
└── LICENSE
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
- unified evaluation datasets,
- cross-model metric comparison,
- qualitative inference visualizations.

Evaluation focuses on:
- precision, recall, and mAP,
- stage-wise performance differences,
- generalization across different aquatic environments.

---

## Research Focus

Key research dimensions explored in this repository include:
- the impact of dataset composition on detection performance,
- trade-offs between model size, accuracy, and efficiency,
- effectiveness of multi-stage training strategies,
- model generalization across heterogeneous visual domains.

---

## Documentation

Each major module contains a dedicated `README.md` describing:
- methodological choices,
- experimental assumptions,
- design decisions and known limitations.

This repository is intended to support:
- academic research and experimentation,
- thesis and technical report development,
- reproducible computer vision workflows.

---


## License

This project is released under the  
**Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)** license.

Commercial use is **not permitted** without explicit permission from the author.
