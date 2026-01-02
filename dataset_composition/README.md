## Dataset composition

This folder contains the notebooks used to **prepare, audit, and merge multiple datasets into a single 
YOLO-ready dataset**.
The workflow assumes that annotations are already (or will be converted to) **YOLO format**, and focuses on 
**data quality, consistency, and reproducibility** before training.

In short, this folder handles:
- dataset inspection and validation
- removal of corrupted or invalid samples
- optional deduplication
- harmonization of labels into a single class (`fish`)
- creation of train / validation / test splits

```text
dataset_composition/
├── 01_annotation_converter.ipynb   # optional: convert annotations to YOLO format
├── 02_datasets_audit.ipynb         # audit, clean,deduplicate, export supervised-ready datasets
├── 03_merge_dataset.ipynb          # merge datasets, unify classes, split train/val/test
└── README
```

### 01_annotation_comnverter.ipynb
Converts and harmonizes all annotations into YOLO-format bounding box coordinates.

### 02_datasets_audit.ipynb
Audits multiple YOLO datasets to ensure data quality before training.  
It scans images and labels, detects corrupted or invalid samples, handles sparse or unlabeled frames correctly, 
optionally removes duplicates, and exports a **clean, supervised-ready version** of each dataset together with CSV reports describing what was kept or removed.

### 03_merge_dataset.ipynb
Merges multiple audited YOLO datasets into a **single unified dataset**. All annotations are harmonized into **one class (`fish`)**, background-only images are preserved. The final dataset is split into **train / validation / test** sets following a stratified strategy.

### Notes on split reproducibility
Minor differences in split sizes compared to the original datasets may occur due to the loss of the original splitting code and the inherent rounding effects of the two-stage stratified sampling process.  
These factors prevent exact replication of sample counts, even when using the same methodology and random seed.

