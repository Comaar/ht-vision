# Bayesian Hyperparameter Optimization

This folder contains Bayesian hyperparameter optimization experiments for the YOLO11 model, implemented using **Optuna**.

The optimization process is structured in **two sequential phases**:

1. **Phase 1** – Core training hyperparameters (optimizer, learning rate, loss weights)
2. **Phase 2** – Data augmentation hyperparameters (geometric and color transforms)

Both phases are executed on the combined dataset.

---

## Contents

| File | Description |
|------|-------------|
| `optuna_phase1_core_hp_search.py` | Phase 1: Core hyperparameter optimization |
| `optuna_phase2_data_augmentation_hp_search.py` | Phase 2: Augmentation optimization |

---

## Usage

### Phase 1: Core Hyperparameters

```bash
python optuna_phase1_core_hp_search.py
```

### Phase 2: Augmentation (after Phase 1)

Update `PHASE1_BEST_PARAMS` in the script with Phase 1 results, then:

```bash
python optuna_phase2_data_augmentation_hp_search.py
```

> **Note:** Update file paths in scripts before running.

---

## Results

### Phase 1 — Best Core Hyperparameters

| Epochs | Optimizer | Dropout | Learning Rate | Box Weight | Class Weight |
|--------|-----------|---------|---------------|------------|--------------|
| 10     | SGD       | 0.10    | 0.00151       | 5.0        | 0.4          |

### Phase 2 — Best Augmentation Parameters

| Epochs | Mosaic | MixUp  | Flip UD | Flip LR | Hue    | Saturation | Value   |
|--------|--------|--------|---------|---------|--------|------------|---------|
| 15     | 0.9129 | 0.4553 | 0.0784  | 0.50    | 0.0083 | 0.0274     | 0.3347  |

