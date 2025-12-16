# Bayesian Hyperparameter Optimization (Optuna)

This folder contains Bayesian hyperparameter optimization experiments for the **final YOLO model**, implemented using **Optuna**.

The optimization process is structured in two sequential phases:

1. **Phase 1.1** – Optimization of core training hyperparameters  
2. **Phase 1.2** – Optimization of data augmentation hyperparameters  

Both phases are executed on the **single-class fish detection dataset**, derived from the Roboflow Aquarium dataset after class merging.

---


---

## Files

- **`optuna_phase1_core_hp_search.py`**  
  Optuna study for optimizing core training hyperparameters such as learning rate, optimizer, dropout, and loss weights.

- **`optuna_phase2_data_augementation_hp_search.py`**  
  Optuna study for optimizing data augmentation hyperparameters, including geometric and color transformations.

---

## Dataset and task definition

- **Dataset source:** Roboflow Aquarium dataset  
- **Task type:** Single-class object detection  
- **Class definition:**  
  The original classes `fish`, `shark`, and `stingray` were merged into a single target class labeled **`fish`**.  
  All other classes were discarded.

This setup simplifies the task to **generic fish detection**, aligning with downstream application requirements.

---

## Results

### Phase 1.1 — Best-performing core hyperparameter configuration

| Epochs | Optimizer | Dropout | Learning rate | Box weight | Class weight |
|--------|-----------|---------|---------------|------------|--------------|
| 10     | SGD       | 0.10    | 0.00151       | 5.0        | 0.4          |

This configuration yielded the best objective value during the Phase 1.1 Optuna study and was selected as the baseline for subsequent optimization.

---

### Phase 1.2 — Best-performing augmentation parameters

| Epochs | Mosaic | MixUp | Flip UD | Flip LR | Hue    | Saturation | Value  |
|--------|--------|-------|---------|---------|--------|------------|--------|
| 15     | 0.9129 | 0.4553| 0.07835 | 0.50    | 0.0083 | 0.02738    | 0.33474|

These augmentation parameters provided the best performance during Phase 1.2, improving robustness through controlled geometric and color transformations.

---

## How to run

From the repository root:

```bash
python final_model_training/bayesian_hp_optimization/optuna_phase1_core_hp_search.py
python final_model_training/bayesian_hp_optimization/optuna_phase2_data_augementation_hp_search.py

