RSNA ICH (1st Place) – Agent Handoff

Purpose
- First‑place RSNA 2019 Intracranial Hemorrhage solution: 2D CNN classifiers + sequence models; DICOM→PNG preprocessing.

Current Status
- Not executed yet. Kaggle credentials present in `~/.kaggle` per user.
- No GPU available; training on CPU will be slow. Prefer using released pretrained models for quick validation.

Environment
- Python with PyTorch (CPU ok for inference). Example pins: `torch==1.12.*` CPU, `opencv-python==3.4.2`, `scikit-image==0.14.0`, `scikit-learn==0.19.1`, `scipy==1.1.0`.

Data
- RSNA competition DICOMs under `stage_1_train_images/`, `stage_2_test_images/`, etc.
- Convert DICOM→PNG using provided scripts:
  - `python3 prepare_data.py -dcm_path stage_1_train_images -png_path train_png`
  - Adjust paths as needed for test sets.

Pretrained Inference (recommended on CPU)
- Download released weights (Google Drive links in README) into the expected folders.
- Predict on sample PNGs:
  - `python3 predict.py -backbone DenseNet121_change_avg -img_size 256 -tbs 4 -vbs 4 -spth DenseNet121_change_avg_256`
- Sequence model:
  - Set paths in `setting.py`, download provided CSV/features, then: `CUDA_VISIBLE_DEVICES=0 python main.py` (for CPU, remove the env var and ensure CPU device is used in code if needed).

Notes
- If CPU inference errors on CUDA calls, patch device handling to support CPU (`.to(device)` pattern) or set PyTorch to CPU only.
- Training from scratch on CPU is not advised; use pretrained checkpoints to validate claims.

AmazonQ – Pick Up Here
- Prepare a small subset of DICOMs, convert to PNGs, download one pretrained model, and run `predict.py` on CPU.
- Record outputs and basic timing; only pursue training after GPU becomes available.

Sync Points
- Shared scratch (create on GPU host): `/lustre/scratch/rsna_ihd/{weights,preds,logs}`
- Helper: `bash bin/create_syncpoints.sh`
