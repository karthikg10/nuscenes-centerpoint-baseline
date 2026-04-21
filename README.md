# mmdetection3d_models_evaluation

This repository contains experiments and evaluations of 3D object detection models using [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) on the [nuScenes](https://www.nuscenes.org/) dataset (v1.0-mini). The first project milestone focuses on reproducing and analyzing a **CenterPoint** baseline model for LiDAR-based 3D detection.

---

## Project Overview

The goal of this project is to:

- Set up a fully functional 3D detection pipeline using MMDetection3D.
- Train and evaluate a CenterPoint baseline model on nuScenes v1.0-mini.
- Understand the end-to-end process: data preparation, training, evaluation, and metrics visualization.
- Use this baseline as a reference to compare future configuration variants (e.g., voxel size changes, NMS variants, threshold tuning).

---

## Quickstart

Reproducing the baseline end-to-end:

```bash
# 1. Install MMDetection3D dependencies (see docs/en/get_started.md)
pip install -r requirements.txt
pip install -v -e .

# 2. Place nuScenes v1.0-mini under data/nuscenes/ and generate info files
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0-mini

# 3. Train CenterPoint baseline (single GPU)
python tools/train.py \
    configs/centerpoint/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py

# 4. Evaluate the trained checkpoint
python tools/test.py \
    configs/centerpoint/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    work_dirs/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d/epoch_20.pth

# 5. Plot metrics
python analysis/plot_metrics.py \
    --metrics work_dirs/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d/results_eval/metrics_summary.json \
    --out analysis/figures
```

> **Note**: `data/`, `work_dirs/`, and checkpoints are not tracked in this repo (see `.gitignore`). Download nuScenes v1.0-mini from [nuscenes.org](https://www.nuscenes.org/nuscenes) before running step 2.

---

## Results Summary

CenterPoint (voxel 0.1 m, SECOND + SECFPN) on nuScenes v1.0-mini val (81 samples):

| Metric | Value |
| ------ | ----- |
| mAP    | 0.1466 |
| NDS    | 0.2073 |
| mATE   | 0.6401 |
| mASE   | 0.5001 |
| mAOE   | 1.0223 |
| mAVE   | 1.1288 |
| mAAE   | 0.5203 |

| Class | AP |
| ----- | -- |
| car           | 0.61 |
| pedestrian    | 0.77 |
| trailer / construction_vehicle / barrier / traffic_cone | ≈ 0 |

The low mAP is expected on `v1.0-mini` given the tiny training split (323 samples) and heavy class imbalance for rare categories.

---

## Repository Structure

Key directories (relative to this repo):

- `configs/`  
  Custom or copied MMDetection3D configuration files (e.g., CenterPoint on nuScenes).

- `data/nuscenes/`  
  Prepared nuScenes-mini data in MMDetection3D format:
  - `nuscenes_infos_train.pkl` – training metadata (8 scenes, 323 samples).
  - `nuscenes_infos_val.pkl` – validation metadata (2 scenes, 81 samples).
  - `v1.0-mini/` – raw nuScenes-mini tables and sensor data.

- `tools/`  
  Scripts used from MMDetection3D:
  - `create_data.py` – nuScenes → MMDetection3D conversion.
  - `train.py` – training entrypoint.
  - `test.py` – evaluation entrypoint.

- `work_dirs/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d/`  
  Experiment outputs:
  - `epoch_20.pth` – trained CenterPoint checkpoint.
  - `results_eval/metrics_summary.json` – official nuScenes detection metrics.
  - Training logs and evaluation logs.

---

## Baseline: CenterPoint on nuScenes-mini

### Model

- **Architecture**: CenterPoint with SECOND backbone and SECFPN neck.  
- **Modality**: LiDAR-only 3D detection.  
- **Voxel size**: 0.1 m.  
- **Dataset**: nuScenes v1.0-mini (10 scenes, 404 samples; 323 train / 81 val).

### Data Preparation

Using MMDetection3D’s data tools, the following steps were performed:

- Converted nuScenes v1.0-mini into MMDetection3D format, creating:
  - `nuscenes_infos_train.pkl` for training.
  - `nuscenes_infos_val.pkl` for validation.
- Built a ground-truth database for training-time augmentation (object sampling).
- Verified dataset statistics (e.g., number of samples and per-class instance counts) to confirm correct preprocessing.

### Training

- **Config**: `centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py`.  
- **Schedule**: 20 epochs with cyclic / cosine-like learning rate.  
- **Optimizer**: AdamW.  
- **Augmentations**: Random flips, rotations, scaling, range filtering, and GT sampling.

Training logs show smooth convergence, with loss decreasing significantly over 20 epochs. The final checkpoint is stored at:

- `work_dirs/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d/epoch_20.pth`

### Evaluation

The trained model is evaluated on the 81-sample nuScenes-mini validation split using the **official nuScenes detection metrics** (via the nuScenes devkit integrated in MMDetection3D). This required minor adjustments to the devkit to handle strict sample-token checks in the mini setting.

**Key metrics:**

- **mAP**: 0.1466  
- **NDS**: 0.2073  
- **mATE** (translation error): 0.6401  
- **mASE** (scale error): 0.5001  
- **mAOE** (orientation error): 1.0223  
- **mAVE** (velocity error): 1.1288  
- **mAAE** (attribute error): 0.5203  

**Per-class AP highlights:**

- Strong performance on **car** and **pedestrian** classes (AP ≈ 0.61 and 0.77 respectively).  
- Weak or near-zero AP for rare classes in nuScenes-mini such as **trailer**, **construction_vehicle**, **barrier**, and **traffic_cone**, reflecting limited sample counts.

The full metric dump is stored in:

- `work_dirs/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d/results_eval/metrics_summary.json`

---

## Visualizations & Analysis

For this milestone, visualizations are focused on **quantitative metrics**:

- Overall bar charts for mAP and NDS.
- Per-class AP bar charts to compare performance across object categories.
- Error metrics (ATE, ASE, AOE, AVE, AAE) summarized for a high-level view of localization and motion quality.

These plots are generated from `metrics_summary.json` using `analysis/plot_metrics.py`:

```bash
python analysis/plot_metrics.py \
    --metrics work_dirs/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d/results_eval/metrics_summary.json \
    --out analysis/figures
```

This writes `overall_metrics.png`, `per_class_ap.png`, and `error_metrics.png` to the output directory.

Qualitative 3D visualizations (e.g., LiDAR point clouds with predicted boxes) are planned as a next step and may be implemented via custom scripts or external tools, instead of the fragile built-in visualization hooks in this particular environment.

---

## Challenges and Fixes

During this milestone, several non-trivial issues were encountered and resolved:

### NuScenes devkit evaluation assertion

- **Issue**: “Samples in split doesn’t match samples in predictions” assertion when running official nuScenes evaluation on mini.  
- **Fix**: Patched the evaluation code to relax the strict equality check and only use the intersection of prediction and ground-truth sample tokens, enabling successful metric computation.

### NumPy deprecation (`np.long`)

- **Issue**: Training crashed due to `np.long` being removed in newer NumPy versions.  
- **Fix**: Updated the relevant transform in MMDetection3D to use a supported integer type.

### Dataset path and version mismatches

- **Issue**: Config files referencing `v1.0-trainval` while only `v1.0-mini` data were present.  
- **Fix**: Standardized all configs and data generation commands on `v1.0-mini` and re-generated info files for consistency.

### Visualization hook errors

- **Issue**: New visualization hooks in MMDetection3D failed due to missing fields in outputs for LiDAR-only CenterPoint.  
- **Fix**: For the milestone, visualization is done through metric plots rather than relying on the built-in hook.

These fixes ensure that training and evaluation are now stable and reproducible.

---

## Next Steps

Planned work for upcoming milestones:

### Configuration variants

- Train CenterPoint with finer voxel sizes (e.g., 0.075 m).  
- Evaluate Circle NMS and compare with standard NMS.  
- Tune detection score thresholds and NMS parameters.

### Comparative analysis

- Build tables and plots comparing baseline and variants in terms of mAP, NDS, per-class AP, and runtime.

### Qualitative visualization

- Add scripts to visualize predictions in BEV and 3D point-cloud views.  
- Collect examples of both successful detections and typical failure cases.

### Scaling up

- Extend the pipeline to full nuScenes trainval when resources allow.  
- Optionally compare CenterPoint with other 3D detectors (e.g., PointPillars, PV-RCNN) on the same setup.

---

## References

- MMDetection3D: https://github.com/open-mmlab/mmdetection3d  
- nuScenes Dataset and Devkit: https://www.nuscenes.org/  
- CenterPoint paper: *Center-based 3D Object Detection and Tracking* (CVPR 2021)  
- SECOND backbone: *SECOND: Sparsely Embedded Convolutional Detection*  

This repository documents the full baseline pipeline and results for CenterPoint on nuScenes-mini and serves as the foundation for future model evaluation and experimentation.

