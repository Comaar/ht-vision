from ultralytics import YOLO
from IPython.display import display, Image
import os
import pandas as pd
import datetime

# ---------------------- Ensure Working Directory ---------------------- #
target_dir = "/mnt/Data1/mpiccolo/Yolo_test"
try:
    os.chdir(target_dir)
    print(f"✅ Changed working directory to: {os.getcwd()}")
except Exception as e:
    print(f"❌ Failed to change directory to {target_dir}: {e}")
    exit(1)

# ---------------------- Config ---------------------- #
models_to_compare = [
    "yolo11m.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolov8m.pt",
    "yolov8_OzFish+AquaCoop.pt"
]

data_yaml_path = os.path.join(target_dir, "dataset", "roboflow_Aquarium_Combined.v6i.yolov8", "data.yaml")
project_name = os.path.join(target_dir, "YOLO_5Models_Comparison_v2")
os.makedirs(project_name, exist_ok=True)

training_log = []

print(f"📊 Starting training for {len(models_to_compare)} models.\n")

# ---------------------- Training Loop ---------------------- #
for model_name in models_to_compare:
    print("\n" + "=" * 70)
    print(f"🚀 TRAINING MODEL: {model_name}")
    print("=" * 70 + "\n")

    run_name_safe = os.path.basename(model_name).replace('.pt', '').replace('+', '_').replace(' ', '_')
    run_name = f"Train_{run_name_safe}"
    run_dir = os.path.join(project_name, run_name)

    # Paths to possible checkpoints
    best_path = os.path.join(run_dir, "weights", "best.pt")
    last_path = os.path.join(run_dir, "weights", "last.pt")
    results_path = os.path.join(run_dir, "results.png")

    # --- Skip if training already completed ---
    if os.path.exists(best_path) and os.path.exists(results_path):
        print(f"✅ Skipping {model_name} — training already completed.")
        training_log.append({
            "model": model_name,
            "train_run": run_name,
            "run_directory": run_dir,
            "status": "skipped_already_trained",
            "timestamp": datetime.datetime.now().isoformat()
        })
        continue

    # --- Resume or start new training ---
    if os.path.exists(last_path):
        model_path = last_path
        print(f"🔁 Resuming training from checkpoint: {last_path}")
    else:
        model_path = model_name
        print(f"🆕 Starting new training from model: {model_path}")

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[ERROR] Could not load model {model_path}. Skipping.\n{e}")
        continue

    try:
        model.train(
            data=data_yaml_path,
            epochs=100,
            imgsz=640,
            batch=16,
            device=0,
            project=project_name,
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
            "train_run": run_name,
            "run_directory": run_dir,
            "resumed_from": model_path if os.path.exists(last_path) else "original .pt",
            "status": "trained/resumed",
            "timestamp": datetime.datetime.now().isoformat()
        })

        # Display results chart (if in Jupyter)
        results_chart_path = os.path.join(run_dir, 'results.png')
        if os.path.exists(results_chart_path):
            try:
                print(f"\n📈 Training Curves for {model_name}:")
                display(Image(filename=results_chart_path))
            except:
                print(f"[NOTE] Not in Jupyter — skipping image display.")
        else:
            print(f"[WARNING] No results chart found at {results_chart_path}.")

    except Exception as e:
        print(f"[ERROR] Training failed for {model_name}. Skipping.\n{e}")
        continue

# ---------------------- Save Training Log ---------------------- #
if training_log:
    df_log = pd.DataFrame(training_log)
    log_csv_path = os.path.join(project_name, "training_runs_log.csv")
    df_log.to_csv(log_csv_path, index=False)
    print(f"\n📁 Training run log saved to: {log_csv_path}")
else:
    print("\n⚠️ No models trained successfully. Check logs for details.")

print("\n🏁 All training complete.")
