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

## Dataset preprocessing and class merging

The original Roboflow Aquarium dataset provides multiple object classes:

- fish
- shark
- stingray
- jellyfish
- penguin
- puffin
- starfish

For this study, a **single-class detection setup** was adopted.  
Annotations belonging to `fish`, `shark`, and `stingray` were merged into one unified class labeled `fish`. All other classes were discarded.

This choice simplifies the task to **generic fish detection**, aligning the model objective with downstream application requirements.

## Models compared

- yolo11m.pt
- yolo11n.pt
- yolo11s.pt
- yolov8m.pt
- yolov8_OzFish+AquaCoop.pt

## Model Performance Comparison

| Model                  | mAP50-95 | mAP50 | Precision | Recall | F1-Score |
|------------------------|----------|-------|-----------|--------|----------|
| YOLO11m                | 0.4657   | 0.8097 | 0.8117    | 0.6866 | 0.7439   |
| YOLO11n                | 0.3595   | 0.6977 | 0.7255    | 0.6102 | 0.6629   |
| YOLO11s                | 0.4703   | 0.7499 | 0.7763    | 0.6588 | 0.7127   |
| YOLOv8m                | 0.4846   | 0.7901 | 0.7837    | 0.7356 | 0.7589   |
| YOLOv8 OzFish+AquaCoop | 0.3570   | 0.6740 | 0.7421    | 0.6048 | 0.6664   |

## Inference comparison (training sample)

| Original | YOLO11m | YOLO11n |
|---------|---------|---------|
| ![](original_inference_pinksalmon.png) | ![](inference_result_Train_yolo11m.png) | ![](inference_result_Train_yolo11n.png) |

| YOLO11s | YOLOv8m | YOLOv8 OzFish + AquaCoop |
|---------|---------|--------------------------|
| ![](inference_result_Train_yolo11s.png) | ![](inference_result_Train_yolov8m3.png) | ![](inference_result_Train_yolov8_OzFish_AquaCoop.png) |

**Notes**
- Metrics are computed on the same validation set.
- Results are summarized in `evaluation_results.csv`.


## How to run training

From the repository root:

```bash
python model_comparison/yolo_5models_comparison_training_resume.py

