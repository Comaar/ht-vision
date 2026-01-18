# Repository Reorganization Plan

**Repository:** ht-vision  
**Date:** 2026-01-18

---

## Overview

This plan organizes changes into **small, reviewable commits** grouped by functional area. Each commit is atomic and can be reviewed/reverted independently.

---

## Phase 1: Documentation (Commits 1–6)

Low-risk changes that add or improve documentation without affecting code behavior.

### Commit 1: Add project requirements file
**Files:** `requirements.txt` (new)

```
Add requirements.txt with project dependencies

- Ultralytics YOLO
- Optuna
- Pandas
- Pillow
- Jupyter dependencies
```

**Command:** `git commit -m "Add requirements.txt for reproducible environment setup"`

---

### Commit 2: Add CONTRIBUTING.md
**Files:** `CONTRIBUTING.md` (new)

```
Add CONTRIBUTING.md with contribution guidelines

- Development setup instructions
- Code style guidelines
- Pull request process
- Issue reporting guidelines
```

**Command:** `git commit -m "Add CONTRIBUTING.md with contribution guidelines"`

---

### Commit 3: Add SECURITY.md
**Files:** `SECURITY.md` (new)

```
Add SECURITY.md with security policy

- Supported versions
- Reporting vulnerabilities
- Disclosure policy
```

**Command:** `git commit -m "Add SECURITY.md with security policy"`

---

### Commit 4: Add CHANGELOG.md
**Files:** `CHANGELOG.md` (new)

```
Add CHANGELOG.md to track version history

- Initial release documentation
- Keep-a-changelog format
```

**Command:** `git commit -m "Add CHANGELOG.md with initial version history"`

---

### Commit 5: Enhance main README
**Files:** `README.md`

```
Enhance README with installation and usage sections

- Add Prerequisites section
- Add Installation section
- Add Quick Start guide
- Update repository structure
- Add links to sub-module READMEs
```

**Command:** `git commit -m "Enhance README with installation and usage instructions"`

---

### Commit 6: Fix typos and sync sub-module READMEs
**Files:** 
- `dataset_composition/README.md`
- `cross_domain_analysis/README.md`

```
Fix typos and update sub-module READMEs

- Fix "01_annotation_comnverter" typo
- Sync cross_domain_analysis README with actual filenames
```

**Command:** `git commit -m "Fix typos and sync sub-module documentation"`

---

## Phase 2: File Naming Corrections (Commits 7–8)

Fixes typos in filenames. These are **rename-only operations** that do not change code behavior.

### Commit 7: Fix typo in Optuna script filename
**Files:** 
- `model/bayesian_hp_optimization/optuna_phase2_data_augementation_hp_search.py` → `optuna_phase2_data_augmentation_hp_search.py`

```
Fix typo: augementation → augmentation

- Rename optuna_phase2_data_augementation_hp_search.py
```

**Command:** `git commit -m "Fix typo in filename: augementation → augmentation"`

---

### Commit 8: Fix typo in notebook filename
**Files:**
- `model_comparison/ds_aquarium_cobined.ipynb` → `ds_aquarium_combined.ipynb`

```
Fix typo: cobined → combined

- Rename ds_aquarium_cobined.ipynb
```

**Command:** `git commit -m "Fix typo in filename: cobined → combined"`

---

## Phase 3: Naming Standardization (Commits 9–10)

Standardizes naming conventions for consistency.

### Commit 9: Standardize results folder naming
**Files:**
- `model/results/Training_curves.ipynb` → `training_curves.ipynb`

```
Standardize notebook naming to lowercase snake_case

- Rename Training_curves.ipynb → training_curves.ipynb
```

**Command:** `git commit -m "Standardize notebook naming to snake_case"`

---

### Commit 10: Standardize cross-domain notebook naming
**Files:**
- `cross_domain_analysis/prepare_cross_domain_scenarios.ipynb` → `00_prepare_cross_domain_scenarios.ipynb`
- `cross_domain_analysis/yolo11m_cross_domain_training.ipynb` → `03_yolo11m_cross_domain_training.ipynb`
- `cross_domain_analysis/distortion_correlation.ipynb` → `04_distortion_correlation.ipynb`

```
Add numerical prefixes to cross_domain_analysis notebooks

- Establishes clear execution order
- Matches pattern used in dataset_composition/
```

**Command:** `git commit -m "Add numerical prefixes to cross-domain notebooks for execution order"`

---

## Phase 4: Code Documentation (Commits 11–13)

Improves code-level documentation without changing behavior.

### Commit 11: Add module docstrings to Python scripts
**Files:**
- `model_comparison/yolo_5models_comparison_training_resume.py`
- `model/bayesian_hp_optimization/optuna_phase1_core_hp_search.py`
- `model/bayesian_hp_optimization/optuna_phase2_data_augmentation_hp_search.py`

```
Add module-level docstrings to Python scripts

- Document purpose, usage, and dependencies
- Add shebang lines for direct execution
```

**Command:** `git commit -m "Add module docstrings and usage documentation to scripts"`

---

### Commit 12: Improve inline comments consistency
**Files:** (same as Commit 11)

```
Standardize inline comments

- Remove redundant comments
- Improve clarity of complex logic
- Ensure consistent comment style
```

**Command:** `git commit -m "Standardize inline comments for clarity"`

---

### Commit 13: Add configuration section to scripts
**Files:** (same as Commit 11)

```
Refactor hardcoded paths to configuration section

- Group all configurable paths at top of file
- Add clear instructions for customization
- Does not change default behavior
```

**Command:** `git commit -m "Refactor configuration paths for easier customization"`

---

## Phase 5: Cleanup (Commit 14)

Minor cleanup tasks.

### Commit 14: Remove empty placeholder directories
**Files:**
- Remove `dataset_composition/Merged/` (if empty and not needed)
- Or add `.gitkeep` if directory is intentionally preserved

```
Clean up empty directories

- Add .gitkeep to preserve intentional empty directories
- Or remove if no longer needed
```

**Command:** `git commit -m "Clean up empty directories"`

---

## Summary

| Phase | Commits | Risk | Description |
|-------|---------|------|-------------|
| 1. Documentation | 1–6 | 🟢 Low | Add/update markdown files |
| 2. Typo Fixes | 7–8 | 🟢 Low | Rename files with typos |
| 3. Naming | 9–10 | 🟡 Medium | Standardize naming conventions |
| 4. Code Docs | 11–13 | 🟢 Low | Improve code documentation |
| 5. Cleanup | 14 | 🟢 Low | Remove/preserve empty dirs |

---

## Execution Order

1. ✅ **Start with Phase 1** - Documentation changes are safest
2. ⏳ **Phase 2** - Typo fixes after documentation is complete
3. ⏳ **Phase 3** - Naming changes (update any references)
4. ⏳ **Phase 4** - Code documentation improvements
5. ⏳ **Phase 5** - Final cleanup

---

## Notes

### Changes NOT Recommended

The following changes were considered but rejected to preserve external behavior:

1. **Major folder restructuring** - Would break existing references
2. **Renaming main folders** - Would break any external documentation or links
3. **Converting notebooks to Python scripts** - Changes execution paradigm
4. **Removing hardcoded paths** - Would require env vars or config files that change behavior

### Post-Reorganization

After completing these changes:

1. Update any CI/CD pipelines if they reference renamed files
2. Inform collaborators of naming changes
3. Consider setting up pre-commit hooks for future consistency

---

*End of Reorganization Plan*
