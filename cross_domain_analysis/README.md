# Cross-Domain Analysis

This folder contains notebooks for preparing cross-domain scenarios, training models, and analyzing the impact of distortions on model performance.

---

## Contents

```text
cross_domain_analysis/
├── prepare_cross_domain_scenarios.ipynb    # Scenario preparation
├── 01_distortion_analysis.ipynb            # Distortion analysis (part 1)
├── 02_distortion_analysis.ipynb            # Distortion analysis (part 2)
├── yolo11m_cross_domain_training.ipynb     # Cross-domain training
├── distortion_correlation.ipynb            # Correlation analysis
└── README.md
```

---

## Notebooks

### prepare_cross_domain_scenarios.ipynb

Prepares cross-domain evaluation scenarios by organizing datasets and generating the required directory structure and configuration files. This notebook defines how source and target domains are combined for training and evaluation.

### 01_distortion_analysis.ipynb & 02_distortion_analysis.ipynb

Analyze the effect of different image distortions across domains. These notebooks aggregate experimental results and produce quantitative summaries describing how distortions impact model performance in cross-domain settings.

### yolo11m_cross_domain_training.ipynb

Runs cross-domain training experiments using the YOLO11m model. It loads the prepared scenarios, launches training runs, and stores the resulting checkpoints and logs.

### distortion_correlation.ipynb

Studies correlations between distortion severity and performance metrics. This notebook focuses on identifying trends and relationships between distortion types and cross-domain generalization behavior.
