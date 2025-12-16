# optuna_phase1_hp_search.py
from ultralytics import YOLO
import os
import optuna
from optuna.exceptions import TrialPruned

# --- Global Configuration ---
# Path to the dataset's data.yaml file. Update as necessary.
DATA_YAML_PATH = os.path.join(
    '/mnt/Data1/mpiccolo/Yolo_test/bO_Yolo/merged_fish_dataset', 'data.yaml'
)

# Fixed augmentation parameters for Phase 1 to isolate core HP impact.
FIXED_AUGMENTATION_PARAMS = {
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

# Optuna study storage and Ultralytics project directory.
# All Ultralytics runs for this phase will be organized under 'ultralytics_runs'.
OPTUNA_STUDY_BASE_DIR_P1 = "/mnt/Data1/mpiccolo/Yolo_test/bO_Yolo/YOLO11_hpsearch_Optuna_Phase1"

os.makedirs(OPTUNA_STUDY_BASE_DIR_P1, exist_ok=True)
os.makedirs(os.path.join(OPTUNA_STUDY_BASE_DIR_P1, "ultralytics_runs"), exist_ok=True)

# SQLite database URL for Optuna study. Enables persistent storage and resumption.
STORAGE_URL_P1 = f"sqlite:///{os.path.join(OPTUNA_STUDY_BASE_DIR_P1, 'optuna_study.db')}"

# --- Objective Function for Optuna Optimization ---
def objective_phase1(trial: optuna.Trial):
    """
    Defines a single training trial for Optuna to optimize core hyperparameters.
    Maximizes validation mAP@0.5.
    """
    # 1. Suggest hyperparameters for the current trial using Optuna's samplers.
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
    dropout = trial.suggest_float("dropout", 0.0, 0.2, step=0.1) # Discrete steps.
    lr0 = trial.suggest_float("lr0", 1e-4, 5e-2, log=True)       # Log-uniform for learning rate.
    box_weight = trial.suggest_float("box", 5.0, 12.5, step=2.5) # Discrete steps.
    cls_weight = trial.suggest_float("cls", 0.2, 0.6, step=0.2) # Discrete steps.

    # Construct unique experiment name and paths for Ultralytics output.
    exp_name = (
        f"trial_{trial.number}_opt{optimizer}_dr{dropout}_lr{lr0:.5f}"
        f"_box{box_weight}_cls{cls_weight}"
    )
    ultralytics_run_dir = os.path.join(OPTUNA_STUDY_BASE_DIR_P1, "ultralytics_runs")
    ultralytics_exp_path = os.path.join(ultralytics_run_dir, exp_name)
    last_ckpt = os.path.join(ultralytics_exp_path, "weights", "last.pt")

    # Resume Ultralytics training if a checkpoint exists for this trial.
    resume_ckpt = None
    resume_flag = False
    if os.path.isfile(last_ckpt):
        print(f"⏯ Resuming Ultralytics training for trial {trial.number} from {last_ckpt}")
        resume_ckpt = last_ckpt
        resume_flag = True
    else:
        print(f"🚀 Starting new Ultralytics training for trial {trial.number}")

    model = YOLO("yolo11m.pt") # Initialize YOLOv11m model from pretrained weights.

    # 2. Train the model with the suggested and fixed hyperparameters.
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=10,                      
        imgsz=640,                      
        batch=16,                       
        device=0,                       
        project=ultralytics_run_dir,    # Base directory for Ultralytics runs.
        name=exp_name,                  # Specific run directory name for this trial.
        resume=resume_flag,
        model=resume_ckpt,
        seed=42,                        

        # Optimization parameters (from Optuna trial).
        optimizer=optimizer,
        lr0=lr0,
        lrf=0.1,                        # Final learning rate ratio.
        momentum=0.95,
        weight_decay=0.0005,

        # Loss balancing parameters (from Optuna trial).
        box=box_weight,
        cls=cls_weight,
        dfl=2.0,

        # Training tricks.
        cos_lr=True,                    # Cosine learning rate scheduler.
        patience=5,                     # Early stopping patience.
        amp=True,                       # Automatic Mixed Precision for speed.
        cache='disk',                   # Cache images to disk for faster loading on large datasets.
        dropout=dropout,                # Dropout rate (from Optuna trial).

        # Fixed augmentation parameters (from global configuration).
        **FIXED_AUGMENTATION_PARAMS,

        # Detection head configuration.
        max_det=1000,

        # Monitoring and logging.
        plots=True,                     # Generate training plots.
        save_period=-1,                 # Save 'last.pt' checkpoint only.
        verbose=False,                  # Suppress verbose output during training.
    )

    # 3. Report the objective value (validation mAP@0.5) to Optuna.
    try:
        metric = results.box.map50 # Correctly access best validation mAP@0.5.
    except AttributeError:
        print(f"Warning: mAP50 not found via 'results.box.map50' for trial {trial.number}. Returning 0.0.")
        metric = 0.0

    # For full epoch-level pruning, a custom Ultralytics callback reporting intermediate
    # results to `trial.report(metric, step)` and checking `trial.should_prune()` is required.
    # Here, the pruner operates on the final metric of these short runs.
    return metric

# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"Creating/Loading Optuna study from {STORAGE_URL_P1}")
    study_phase1 = optuna.create_study(
        direction="maximize",                      # Objective: maximize mAP.
        sampler=optuna.samplers.TPESampler(seed=42), # TPE algorithm for efficient search.
        pruner=optuna.pruners.MedianPruner(        # Median Pruner for early trial stopping.
            n_startup_trials=5,                    # Do not prune during the first 5 trials.
            n_warmup_steps=3,                      # Do not prune within the first 3 epochs.
            interval_steps=1                       # Check for pruning every step (requires callback).
        ),
        study_name="YOLO11_Phase1_Core_HP_Search",
        storage=STORAGE_URL_P1,
        load_if_exists=True                        # Allow resuming existing study.
    )

    print(f"Starting Optuna Phase 1 optimization. Current trials completed: {len(study_phase1.trials)}")
    n_trials_phase1 = 100 # Total trials for Phase 1. Adjust based on computational budget.

    try:
        study_phase1.optimize(
            objective_phase1,
            n_trials=n_trials_phase1,
            show_progress_bar=True,
            gc_after_trial=True                    # Clean up memory after each trial.
        )
    except Exception as e:
        print(f"An error occurred during Optuna Phase 1 optimization: {e}")
        # Optuna automatically saves progress to DB even if interrupted.

    print("\n--- Optuna Phase 1 Complete ---")
    print("Best trial found:")
    best_trial_p1 = study_phase1.best_trial

    print(f"  Value (mAP50): {best_trial_p1.value:.4f}")
    print("  Best Parameters:")
    for key, value in best_trial_p1.params.items():
        print(f"    {key}: {value}")

    print(f"\nPhase 1 results and Ultralytics runs are in '{OPTUNA_STUDY_BASE_DIR_P1}'.")
    print("\n--- ACTION REQUIRED ---")
    print("1. Analyze the results, noting the 'Best Parameters' above.")
    print("2. Manually update the 'BEST_OPTIMIZER_P1', 'BEST_DROPOUT_P1', etc., variables")
    print("   in 'optuna_phase2_hp_search.py' with these optimal values before running Phase 2.")