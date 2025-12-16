## YOLO11 two-stage training notebook

The notebook `YOLO11_HT_Vision_two_stage_training.ipynb` implements a **two-stage training pipeline** for YOLO11 on the HT_Vision dataset, based on progressive resizing and robust resume/skip logic.

> **Note on stage naming:**  
> Conceptually, the workflow consists of **two stages**:
> 1. an initial training stage at 640×640, and  
> 2. a final fine-tuning stage at higher resolution.  
>
> In the implementation and filesystem, the final fine-tuning stage is still referred to as **Stage 3** (e.g. `Stage3/` directories and variables).  
> This naming is intentionally preserved to remain consistent with the existing Jupyter notebook and previously generated training artifacts.

### Dataset configuration
- Dataset source: Roboflow Aquarium (class-merged).
- Task: single-class object detection (`fish`).
- Dataset paths are defined via absolute paths and must be adapted for other environments.

### Output structure
Training outputs are written to:

- `Training_Results/Stage1` – 640×640 training (initial stage)
- `Training_Results/Stage3` – 1024×1024 fine-tuning (**conceptual Stage 2 / final model**)

Each stage stores standard Ultralytics artifacts (`weights/best.pt`, `weights/last.pt`, `results.csv`).

### Stage definitions
- **Stage 1 (640×640)**  
  Training starts from base YOLO11 weights. If `last.pt` exists, training resumes; if the target number of epochs has already been reached, the stage is skipped.

- **Stage 3 (1024×1024 – final)**  
  Final fine-tuning stage, conceptually corresponding to **Stage 2** in the two-stage pipeline.  
  Training starts from the best checkpoint produced by Stage 1 and produces the final high-resolution model.

