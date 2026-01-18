#!/usr/bin/env python3
"""
Multi-model YOLO comparison training script.
Supports checkpoint resumption and skip logic for completed runs.
"""

from ultralytics import YOLO
from IPython.display import display, Image
import os
import pandas as pd
import datetime


# ==============================================================================
# Configuration — update these paths before running
# ==============================================================================
WORKING_DIR = os.environ.get("YOLO_WORKDIR", "./runs")
DATASET_PATH = os.environ.get("YOLO_DATA_YAML", "./dataset/data.yaml")
OUTPUT_DIR = os.environ.get("YOLO_OUTPUT_DIR", "./comparison_results")

MODELS = [
    "yolo11m.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolov8m.pt",
    "yolov8_OzFish+AquaCoop.pt",
]

# ==============================================================================
# Setup
# ==============================================================================
try:
    os.chdir(WORKING_DIR)
    print(f"Working directory: {os.getcwd()}")
except Exception as e:
    print(f"[ERROR] Failed to change directory: {e}")
    exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)
training_log = []

print(f"Starting training for {len(MODELS)} models.\n")

# ==============================================================================
# Training loop
# ==============================================================================
for model_name in MODELS:
    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70 + "\n")

    run_name_safe = os.path.basename(model_name).replace('.pt', '').replace('+', '_').replace(' ', '_')
    run_name = f"Train_{run_name_safe}"
    run_dir = os.path.join(OUTPUT_DIR, run_name)

    best_path = os.path.join(run_dir, "weights", "best.pt")
    last_path = os.path.join(run_dir, "weights", "last.pt")
    results_path = os.path.join(run_dir, "results.png")

    # Skip if already trained
    if os.path.exists(best_path) and os.path.exists(results_path):
        print(f"Skipping {model_name} — already completed.")
        training_log.append({
            "model": model_name,
            "run": run_name,
            "run_directory": run_dir,
            "status": "skipped",
            "timestamp": datetime.datetime.now().isoformat()
        })
        continue

    # Resume from checkpoint if available
    resumed = False
    if os.path.exists(last_path):
        model_path = last_path
        resumed = True
        print(f"Resuming from checkpoint: {last_path}")
    else:
        model_path = model_name
        print(f"Starting fresh training: {model_path}")

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        continue

    try:
        model.train(
            data=DATASET_PATH,
            epochs=100,
            imgsz=640,
            batch=16,
            device=0,
            project=OUTPUT_DIR,
            name=run_name,
            patience=10,
            optimizer='SGD',
            lr0=0.001,
            warmup_epochs=5.0,
            degrees=10.0,
            translate=0.15,
            scale=0.6,
            hsv_h=0.020,
            hsv_s=0.8
        )

        training_log.append({
            "model": model_name,
            "run": run_name,
            "run_directory": run_dir,
            "resumed_from": model_path if resumed else "original",
            "status": "completed",
            "timestamp": datetime.datetime.now().isoformat()
        })

        # Show results if in Jupyter
        if os.path.exists(results_path):
            try:
                print(f"\nTraining curves for {model_name}:")
                display(Image(filename=results_path))
            except Exception:
                pass

    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        continue

# ==============================================================================
# Save log
# ==============================================================================
if training_log:
    df = pd.DataFrame(training_log)
    log_path = os.path.join(OUTPUT_DIR, "training_log.csv")
    df.to_csv(log_path, index=False)
    print(f"\nLog saved: {log_path}")
else:
    print("\nNo models trained successfully. Check logs for details.")

print("\nAll training complete.")
