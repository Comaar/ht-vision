# optuna_phase2_hp_search.py
import os
import optuna
from ultralytics import YOLO


# ==============================================================================
# Configuration — update these paths before running
# ==============================================================================
WEIGHTS_PATH = os.environ.get("YOLO_WEIGHTS", "yolo11m.pt")
DATA_YAML = os.environ.get("YOLO_DATA_YAML", "./dataset/data.yaml")
OUTPUT_DIR = os.environ.get("OPTUNA_OUTPUT_DIR", "./optuna_phase2")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "runs"), exist_ok=True)

STORAGE_URL = f"sqlite:///{OUTPUT_DIR}/study.db"
RUN_DIR = os.path.join(OUTPUT_DIR, "runs")

# Best params from Phase 1 (update after running phase 1)
PHASE1_BEST = {
    "optimizer": "SGD",
    "dropout": 0.1,
    "lr0": 0.0015124195296756932,
    "box": 5.0,
    "cls": 0.4
}

print("\n--- Phase 1 Parameters ---")
for k, v in PHASE1_BEST.items():
    print(f"  {k}: {v}")
print("--------------------------\n")


def objective(trial):
    """Single trial: train with suggested augmentation params, return 1-mAP@0.5."""
    
    lr0 = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)
    mosaic = trial.suggest_float("mosaic", 0.0, 1.0)
    mixup = trial.suggest_float("mixup", 0.0, 0.5)
    flipud = trial.suggest_float("flipud", 0.0, 0.5)
    hsv_h = trial.suggest_float("hsv_h", 0.0, 0.1)
    hsv_s = trial.suggest_float("hsv_s", 0.0, 1.0)
    hsv_v = trial.suggest_float("hsv_v", 0.0, 1.0)

    try:
        model = YOLO(WEIGHTS_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise optuna.exceptions.TrialPruned()

    trial_name = f"t{trial.number}_mos{mosaic:.3f}_mix{mixup:.3f}_lr{lr0:.4f}"

    try:
        results = model.train(
            data=DATA_YAML,
            epochs=15,
            batch=16,
            lr0=lr0,
            optimizer=PHASE1_BEST["optimizer"],
            dropout=PHASE1_BEST["dropout"],
            box=PHASE1_BEST["box"],
            cls=PHASE1_BEST["cls"],
            mosaic=mosaic,
            mixup=mixup,
            flipud=flipud,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            project=RUN_DIR,
            name=trial_name,
            exist_ok=True,
            verbose=False,
            val=True
        )

        # Extract mAP50
        train_result = results[0] if isinstance(results, list) else results
        mAP50 = 0.0

        if hasattr(train_result, "metrics") and isinstance(train_result.metrics, dict):
            for key in ["metrics/mAP50(B)", "mAP50"]:
                if key in train_result.metrics:
                    mAP50 = train_result.metrics[key]
                    break
        elif hasattr(train_result, "box") and hasattr(train_result.box, "map50"):
            mAP50 = train_result.box.map50

        return 1.0 - mAP50

    except Exception as e:
        print(f"Training failed for trial {trial.number}: {e}")
        raise optuna.exceptions.TrialPruned()


# ==============================================================================
# Create or load study
# ==============================================================================
study = optuna.create_study(
    study_name="phase2_augmentation",
    direction="minimize",
    storage=STORAGE_URL,
    load_if_exists=True
)


def count_complete(s):
    return sum(1 for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE)


initial_complete = count_complete(study)
print(f"Loaded study. Completed trials: {initial_complete}")

TARGET_TRIALS = 70
MAX_ATTEMPTS = 200

# ==============================================================================
# Optimization loop
# ==============================================================================
current_complete = initial_complete

if current_complete >= TARGET_TRIALS:
    print(f"Target already reached: {current_complete} trials.")
else:
    print(f"Need {TARGET_TRIALS - current_complete} more trials.")

    while current_complete < TARGET_TRIALS and len(study.trials) < MAX_ATTEMPTS:
        batch = min(10, TARGET_TRIALS - current_complete + 5, MAX_ATTEMPTS - len(study.trials))
        if batch <= 0:
            break

        print(f"\nRunning batch of {batch} trials...")
        try:
            study.optimize(objective, n_trials=batch, show_progress_bar=True)
        except Exception as e:
            print(f"Optimization error: {e}")
            break

        current_complete = count_complete(study)
        print(f"Completed: {current_complete}")

    print(f"\nFinal: {current_complete}/{TARGET_TRIALS} trials")

# ==============================================================================
# Results
# ==============================================================================
print("\n=== Phase 2 Results ===")
if study.best_trial and study.best_trial.state == optuna.trial.TrialState.COMPLETE:
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (1 - mAP50): {study.best_trial.value:.4f}")
    print("Parameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
else:
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if complete:
        best = min(complete, key=lambda t: t.value)
        print(f"Best trial: {best.number}")
        print(f"Best value: {best.value:.4f}")
        for k, v in best.params.items():
            print(f"  {k}: {v}")
    else:
        print("No completed trials found.")
