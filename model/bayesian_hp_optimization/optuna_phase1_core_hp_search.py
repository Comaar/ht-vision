# optuna_phase1_hp_search.py
from ultralytics import YOLO
import os
import optuna
from optuna.exceptions import TrialPruned


# ==============================================================================
# Configuration — update these paths before running
# ==============================================================================
DATA_YAML_PATH = os.environ.get("YOLO_DATA_YAML", "./dataset/data.yaml")
OUTPUT_DIR = os.environ.get("OPTUNA_OUTPUT_DIR", "./optuna_phase1")
BASE_MODEL = os.environ.get("YOLO_BASE_MODEL", "yolo11m.pt")

# Fixed augmentation params (not optimized in phase 1)
FIXED_AUG = {
    'mosaic': 0.8,
    'mixup': 0.2,
    'flipud': 0.5,
    'copy_paste': 0.1,
    'scale': 0.5,
    'shear': 2.0,
    'hsv_h': 0.02,
    'hsv_s': 0.8,
    'hsv_v': 0.4,
}

# Setup directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "runs"), exist_ok=True)

STORAGE_URL = f"sqlite:///{os.path.join(OUTPUT_DIR, 'study.db')}"


def objective(trial: optuna.Trial):
    """Single trial: train with suggested hyperparameters, return mAP@0.5."""
    
    # Sample hyperparameters
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
    dropout = trial.suggest_float("dropout", 0.0, 0.2, step=0.1)
    lr0 = trial.suggest_float("lr0", 1e-4, 5e-2, log=True)
    box_weight = trial.suggest_float("box", 5.0, 12.5, step=2.5)
    cls_weight = trial.suggest_float("cls", 0.2, 0.6, step=0.2)

    exp_name = f"t{trial.number}_opt{optimizer}_dr{dropout}_lr{lr0:.5f}_box{box_weight}_cls{cls_weight}"
    run_dir = os.path.join(OUTPUT_DIR, "runs")
    exp_path = os.path.join(run_dir, exp_name)
    last_ckpt = os.path.join(exp_path, "weights", "last.pt")

    # Check for resumption
    resume_flag = os.path.isfile(last_ckpt)
    if resume_flag:
        print(f"Resuming trial {trial.number} from checkpoint")
    else:
        print(f"Starting trial {trial.number}")

    model = YOLO(BASE_MODEL)

    results = model.train(
        data=DATA_YAML_PATH,
        epochs=10,
        imgsz=640,
        batch=16,
        device=0,
        project=run_dir,
        name=exp_name,
        resume=resume_flag,
        model=last_ckpt if resume_flag else None,
        seed=42,
        optimizer=optimizer,
        lr0=lr0,
        lrf=0.1,
        momentum=0.95,
        weight_decay=0.0005,
        box=box_weight,
        cls=cls_weight,
        dfl=2.0,
        cos_lr=True,
        patience=5,
        amp=True,
        cache='disk',
        dropout=dropout,
        max_det=1000,
        plots=True,
        save_period=-1,
        verbose=False,
        **FIXED_AUG,
    )

    try:
        metric = results.box.map50
    except AttributeError:
        print(f"Warning: mAP50 not found for trial {trial.number}, returning 0.")
        metric = 0.0

    return metric


if __name__ == "__main__":
    print(f"Study storage: {STORAGE_URL}")
    
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1),
        study_name="phase1_core_hp",
        storage=STORAGE_URL,
        load_if_exists=True
    )

    print(f"Completed trials: {len(study.trials)}")
    n_trials = 100

    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True, gc_after_trial=True)
    except Exception as e:
        print(f"Optimization error: {e}")

    print("\n--- Phase 1 Complete ---")
    if len(study.trials) > 0 and study.best_trial is not None:
        best = study.best_trial
        print(f"Best mAP50: {best.value:.4f}")
        print("Best parameters:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")
    else:
        print("No completed trials found.")
