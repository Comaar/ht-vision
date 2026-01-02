# Dataset Composition

Tools for auditing, validating, and composing datasets used in the 
ht-vision project.

## Contents

- `Step3_Merge_Datasets_GitHub.ipynb`  
  Merge multiple datasets into a unified structure with reproducible paths
  and optional validation steps.

- `Datasets_Audit_copia_github_polished.ipynb`  
  Inspect dataset structure, labels, and basic consistency before 
training.

## Notes

- No hard-coded local paths
- Configuration via environment variables where applicable
- Intended to be run before model training
- Minor differences in split sizes compared to the original thesis are due 
to the loss of the original code and the inherent rounding effects of 
two-stage stratified sampling, which prevent exact replication of split 
counts despite identical methodology and random seed.

