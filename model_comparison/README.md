# Model Comparison (YOLO)

This folder contains code and results for comparing multiple YOLO models on the Aquarium dataset.

## Contents

- `yolo_5models_comparison_training_resume.py`  
  Training script that can skip completed runs and resume from checkpoints (`last.pt`) when available.

- `evaluation_results.csv`  
  Summary metrics for each trained model (mAP50-95, mAP50, Precision, Recall, F1).

- `ds_aquarium_cobined.ipynb`  
  Notebook used for dataset inspection / preparation (as used on the training server).

- `configs/data.yaml`  
  Dataset configuration used by Ultralytics YOLO.

## Models compared

- yolo11m.pt
- yolo11n.pt
- yolo11s.pt
- yolov8m.pt
- yolov8_OzFish+AquaCoop.pt

## How to run training

From the repository root:

```bash
python model_comparison/yolo_5models_comparison_training_resume.py

