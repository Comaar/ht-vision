# Cross-Domain Analysis

This folder contains the notebooks used to prepare cross-domain scenarios, 
train models, and analyze the impact of distortions on performance.

## Contents

### `prepare_cross_domain_scenarios.ipynb`
Prepares cross-domain evaluation scenarios by organizing datasets and 
generating the required directory structure and configuration files.  
This notebook is responsible for defining how source and target domains are 
combined for training and evaluation.

### `yolo11m_cross_domain_training.ipynb`
Runs cross-domain training experiments using the YOLO11m model.  
It loads the prepared scenarios, launches training runs, and stores the 
resulting checkpoints and logs.  
The notebook is designed to be portable and relies on relative paths or 
environment variables.

### `distortion_analysis.ipynb`
Analyzes the effect of different image distortions across domains.  
It aggregates experimental results and produces quantitative summaries 
describing how distortions impact model performance in cross-domain settings.

### `distortion_correlation.ipynb`
Studies correlations between distortion severity and performance metrics.  
This notebook focuses on identifying trends and relationships between 
distortion types and cross-domain generalization behavior.

## Notes
- All notebooks avoid hard-coded machine-specific paths and are intended to 
run in a portable environment.
- Expected data and experiment outputs should be placed in the appropriate 
relative directories before execution.

