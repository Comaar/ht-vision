# optuna_phase2_hp_search.py
import os
import optuna
from ultralytics import YOLO

# -----------------------------
# Paths and constants
# -----------------------------
WEIGHTS_PATH = "/mnt/Data1/mpiccolo/Yolo_test/bO_Yolo/yolo11m.pt"
DATA_YAML = "/mnt/Data1/mpiccolo/Yolo_test/bO_Yolo/merged_fish_dataset/data.yaml"

# --- MODIFIED FOR RESUMPTION ---
# This path MUST point to your *existing* Optuna database directory.
# Based on your previous output, this is the correct path.
OPTUNA_BASE_DIR = "/mnt/Data1/mpiccolo/Yolo_test/bO_Yolo/YOLO11_hpsearch_Optuna_Phase2"
OPTUNA_DB_PATH = f"sqlite:///{OPTUNA_BASE_DIR}/optuna_study.db"
ULTRALYTICS_RUNS = f"{OPTUNA_BASE_DIR}/ultralytics_runs"
# -------------------------------

# Ensure the directories exist. os.makedirs with exist_ok=True will not overwrite.
os.makedirs(OPTUNA_BASE_DIR, exist_ok=True)
os.makedirs(ULTRALYTICS_RUNS, exist_ok=True)

# -----------------------------
# Phase 1 best parameters
# -----------------------------
PHASE1_BEST_PARAMS = {
    "optimizer": "SGD",
    "dropout": 0.1,
    "lr0": 0.0015124195296756932,
    "box": 5.0,
    "cls": 0.4
}

print("\n--- Phase 1 Best Parameters Loaded ---")
for k, v in PHASE1_BEST_PARAMS.items():
    print(f"  {k.capitalize()}: {v}")
print("------------------------------------\n")

# -----------------------------
# Optuna objective function
# -----------------------------
def objective(trial):
    # Sample hyperparameters for augmentation and learning rate
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
        print(f"❌ Failed to load YOLO model from {WEIGHTS_PATH}: {e}")
        # Prune trial if model loading fails
        raise optuna.exceptions.TrialPruned()
    
    # Construct a unique name for the Ultralytics run based on sampled hyperparameters
    trial_name = (
        f"trial_mos{mosaic:.4f}_mix{mixup:.4f}_fud{flipud:.4f}_hh{hsv_h:.4f}_hs{hsv_s:.4f}_hv{hsv_v:.4f}_opt{PHASE1_BEST_PARAMS['optimizer']}_lr{lr0:.4f}"
    )
    
    try:
        # Perform training with current hyperparameters
        results = model.train(
            data=DATA_YAML,
            epochs=15, # As defined in your original code
            batch=16,   # As defined in your original code
            lr0=lr0,
            optimizer=PHASE1_BEST_PARAMS["optimizer"],
            dropout=PHASE1_BEST_PARAMS["dropout"],
            box=PHASE1_BEST_PARAMS["box"],
            cls=PHASE1_BEST_PARAMS["cls"],
            mosaic=mosaic,
            mixup=mixup,
            flipud=flipud,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            project=ULTRALYTICS_RUNS, # Ultralytics runs will be saved here
            name=trial_name,         # Unique name for this trial's Ultralytics run
            exist_ok=True,           # Allow existing run directories (though unique name minimizes this)
            verbose=False,           # Suppress excessive Ultralytics output to console
            val=True                 # Perform validation
        )

        # Access mAP50 safely from the results object
        # Ultralytics results can be a single object or a list.
        train_result = results[0] if isinstance(results, list) else results
        mAP50 = 0.0 # Default value if mAP50 cannot be found

        if hasattr(train_result, "metrics") and isinstance(train_result.metrics, dict):
            # Ultralytics 8+ typically stores metrics in a dictionary under `metrics` attribute
            # Common keys are 'metrics/mAP50(B)', 'mAP50', 'metrics/mAP50-95(B)'
            mAP50_key_options = ["metrics/mAP50(B)", "mAP50"]
            for key in mAP50_key_options:
                if key in train_result.metrics:
                    mAP50 = train_result.metrics[key]
                    break
            else:
                print(f"Warning: 'mAP50' or 'metrics/mAP50(B)' not found in train_result.metrics for trial {trial.number}. Available keys: {train_result.metrics.keys()}. Using default 0.")
        elif hasattr(train_result, "box") and hasattr(train_result.box, "map50"):
            # Older Ultralytics versions or direct access to box metrics
            mAP50 = train_result.box.map50
        else:
            print(f"Warning: Could not find mAP50 in results for trial {trial.number}. Returning 0.")
            mAP50 = 0.0
            
        # Optuna's direction is 'minimize', so we return 1.0 - mAP50
        return 1.0 - mAP50

    except Exception as e:
        print(f"❌ Training failed for trial {trial.number}: {e}")
        # Prune the trial if training fails to avoid considering invalid results
        raise optuna.exceptions.TrialPruned()


# -----------------------------
# Create or load Optuna study
# -----------------------------
# IMPORTANT: Use the correct study name identified from your database inspection:
# "YOLO11_Phase2_Aug_HP_Search"
study_name = "YOLO11_Phase2_Aug_HP_Search" 
study = optuna.create_study(
    study_name=study_name,
    direction="minimize", # We are minimizing (1 - mAP50)
    storage=OPTUNA_DB_PATH,
    load_if_exists=True   # This is crucial for resuming an existing study
)

# -----------------------------
# Helper to count successful trials
# -----------------------------
def count_successful_trials(optuna_study):
    """Counts the number of trials with state COMPLETE."""
    return sum(1 for t in optuna_study.trials if t.state == optuna.trial.TrialState.COMPLETE)

# -----------------------------
# Determine current successful trials and define target
# -----------------------------
initial_successful_trials = count_successful_trials(study)
print(f"Loaded study '{study_name}'. Current total trials in DB: {len(study.trials)}. Successful trials: {initial_successful_trials}")

# Define your target number of *successful* trials (e.g., 70, 100, etc.)
TARGET_TOTAL_SUCCESSFUL_TRIALS = 70 

# Safety net: Set a maximum total number of trials to attempt.
# This prevents an infinite loop if trials consistently fail/prune, 
# exceeding your desired successful trial count. Adjust this as needed.
MAX_TOTAL_TRIALS_TO_ATTEMPT = 200 

print(f"Targeting a total of {TARGET_TOTAL_SUCCESSFUL_TRIALS} successful trials.")

# -----------------------------
# Run optimization until target successful trials are met
# -----------------------------
current_successful_trials = initial_successful_trials

if current_successful_trials >= TARGET_TOTAL_SUCCESSFUL_TRIALS:
    print(f"Already achieved or exceeded target of {TARGET_TOTAL_SUCCESSFUL_TRIALS} successful trials ({current_successful_trials} successful trials found).")
else:
    # Fix the typo here: TARGET_TOTAL_SUCCESSUAL_TRIALS -> TARGET_TOTAL_SUCCESSFUL_TRIALS
    print(f"Need {TARGET_TOTAL_SUCCESSFUL_TRIALS - current_successful_trials} more successful trials.") 
    
    while current_successful_trials < TARGET_TOTAL_SUCCESSFUL_TRIALS and len(study.trials) < MAX_TOTAL_TRIALS_TO_ATTEMPT:
        # Determine how many new trials to run in this batch.
        # Run at least a small batch (e.g., 5-10 trials) to progress, 
        # or enough to bridge the gap to the target plus a small buffer.
        batch_size = max(10, TARGET_TOTAL_SUCCESSFUL_TRIALS - current_successful_trials + 5) 
        
        # Ensure we don't exceed the MAX_TOTAL_TRIALS_TO_ATTEMPT
        remaining_capacity_for_new_trials = MAX_TOTAL_TRIALS_TO_ATTEMPT - len(study.trials)
        if remaining_capacity_for_new_trials <= 0:
            print(f"Maximum total trials ({MAX_TOTAL_TRIALS_TO_ATTEMPT}) reached. Stopping optimization.")
            break
        
        trials_to_optimize_this_batch = min(batch_size, remaining_capacity_for_new_trials)

        print(f"\nRunning a batch of {trials_to_optimize_this_batch} new trials... (Current total trials in DB: {len(study.trials)})")
        try:
            study.optimize(objective, n_trials=trials_to_optimize_this_batch, show_progress_bar=True)
        except optuna.exceptions.StudyDirectionNotSetError: # Handle if study was created without direction
            print("Error: Study direction not set. This typically happens if the study was created with a very old Optuna version or incorrectly. Please ensure your Optuna version is up-to-date and the study was initialized with a direction.")
            break
        except Exception as e:
            print(f"An unexpected error occurred during study.optimize: {e}")
            # Decide if you want to continue or break based on the error
            break 
            
        current_successful_trials = count_successful_trials(study)
        print(f"Batch completed. New total successful trials: {current_successful_trials}")

    print(f"\nOptimization loop finished. Final successful trials: {current_successful_trials} / {TARGET_TOTAL_SUCCESSFUL_TRIALS}")
    if current_successful_trials < TARGET_TOTAL_SUCCESSFUL_TRIALS:
        print(f"Warning: Could not reach the target of {TARGET_TOTAL_SUCCESSFUL_TRIALS} successful trials.")
    if len(study.trials) >= MAX_TOTAL_TRIALS_TO_ATTEMPT:
        print(f"Note: Optimization stopped because MAX_TOTAL_TRIALS_TO_ATTEMPT ({MAX_TOTAL_TRIALS_TO_ATTEMPT}) was reached.")


# -----------------------------
# Best hyperparameters
# -----------------------------
print("\n=== Phase 2 Best Hyperparameters ===")
# Ensure there is at least one complete trial before accessing best_trial
if study.best_trial and study.best_trial.state == optuna.trial.TrialState.COMPLETE:
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best value (1 - mAP50): {study.best_trial.value:.4f}")
    print("Best parameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
else:
    # If no complete trials, best_trial might be None or not COMPLETE
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if complete_trials:
        # Fallback: manually find the best among complete trials if study.best_trial isn't robust
        best_complete_trial = min(complete_trials, key=lambda t: t.value)
        print(f"Best COMPLETE trial (found manually): {best_complete_trial.number}")
        print(f"Best value (1 - mAP50): {best_complete_trial.value:.4f}")
        print("Best parameters:")
        for k, v in best_complete_trial.params.items():
            print(f"  {k}: {v}")
    else:
        print("No successful (COMPLETE) trials were found to determine the best hyperparameters.")