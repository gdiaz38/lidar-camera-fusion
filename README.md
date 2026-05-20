# 🚗 LiDAR–Camera Sensor Fusion for ADAS

A complete multi-sensor fusion pipeline that combines 3D LiDAR point clouds with 2D camera images to detect, range, and track objects in real driving scenes. Built on the KITTI dataset as part of an ADAS portfolio project.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-live-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📊 Live Dashboard

👉 **[View Live App](https://gdiaz38-lidar-camera-fusion.streamlit.app)**

---

## Overview

Modern ADAS systems depend on fusing multiple sensors because no single sensor is sufficient alone — cameras provide rich semantic information but no depth; LiDAR provides precise 3D geometry but no class labels. This project implements a complete late-fusion pipeline that combines both, then tracks detected objects across frames.

Key question it answers: *What objects are in the scene, how far away are they, and where are they moving?*

---

## Key Results

| Metric | Value |
|---|---|
| Frames processed | 7,481 training frames |
| Max simultaneous tracks | **6 confirmed tracks** |
| Detection range | **6–78 meters** |
| Ghost tracks suppressed | **0** |
| Classes detected | Car · Pedestrian · Cyclist |

---

## Features

- **Full results gallery** — 20 fusion and tracking output frames across diverse KITTI scenes
- **Frame-by-frame playback** — simulated pipeline replay showing track initialization, confirmation, and range updates
- **Tracking metrics** — confirmed track count, closest/farthest object, distance range per frame
- **Pipeline architecture walkthrough** — all 6 stages with Kalman Filter state vector documentation
- **Distance distribution** — histogram of detected object ranges across the sequence

---

## Pipeline Architecture

```
LiDAR (.bin) ──► Point Cloud Projection ──────────────────┐
                  (calibration matrices)                   ▼
Camera (.png) ──► YOLOv8 Detection ──► Frustum Extraction ──► Fused Detections
                  (bounding boxes)     (depth per box)         (class + distance)
                                                               │
                                                               ▼
                                                    Kalman Filter Tracker
                                                    (persistent IDs across frames)
```

---

## Pipeline Stages

| Stage | File | Description |
|---|---|---|
| 01 | `src/calibration.py` | Parse KITTI calib files → P2, R0, Tr matrices |
| 02 | `src/projection.py` | Project N×3 LiDAR cloud into image space |
| 03 | `notebooks/03_yolo_detection.ipynb` | YOLOv8 2D detection on camera frames |
| 04 | `src/projection.py` | Frustum extraction — LiDAR depth per bounding box |
| 05 | `notebooks/05_fusion_pipeline.ipynb` | Fuse class label + depth → ranged detection |
| 06 | `src/tracker.py` | Kalman Filter + Hungarian multi-object tracker |

---

## Project Structure

```
lidar-camera-fusion/
├── dashboard.py              # Streamlit results showcase
├── src/
│   ├── calibration.py        # KITTI calibration matrix parsing
│   ├── projection.py         # LiDAR → image projection + frustum extraction
│   └── tracker.py            # Kalman Filter + Hungarian multi-object tracker
├── notebooks/
│   ├── 01_calibration.ipynb
│   ├── 02_lidar_projection.ipynb
│   ├── 03_yolo_detection.ipynb
│   ├── 04_frustum_extraction.ipynb
│   ├── 05_fusion_pipeline.ipynb
│   └── 06_kalman_tracking.ipynb
├── results/                  # 20 output frames (fusion + tracking)
└── requirements.txt
```

---

## How It Works

### Stage 1 — Sensor Calibration
Parse KITTI calibration files to extract three matrices:
- **P2** — 3×4 camera projection matrix
- **R0** — 3×3 rectification matrix
- **Tr** — 3×4 LiDAR-to-camera extrinsic transform

### Stage 2 — LiDAR Projection
Transform each 3D point from the velodyne frame into image space:
```
pts_cam = R0_full @ Tr_full @ pts_hom.T
pts_img = P2 @ pts_cam
pixels  = pts_img[:2] / pts_img[2]   # normalize by depth
```
Filter points behind the camera (depth ≤ 0).

### Stage 3 — YOLOv8 Detection
Run YOLOv8n on the camera image to produce 2D bounding boxes with class labels and confidence scores.

### Stage 4 — Frustum Extraction
For each 2D bounding box, extract all LiDAR points that project inside it. Compute robust depth estimate:
```python
median = np.median(depth_in_box)
clean  = depth_in_box[np.abs(depth_in_box - median) < 2.0]
distance = clean.mean()
```

### Stage 5 — Sensor Fusion
Combine YOLOv8 label + LiDAR depth → fused detection: `{class, confidence, distance_m, bbox_2d}`.

### Stage 6 — Kalman Tracking
SORT-style multi-object tracker:
- **State vector:** `[cx, cy, w, h, ẋ, ẏ, ẇ, ḣ]` — 8D constant-velocity model
- **Assignment:** Hungarian algorithm on IoU cost matrix (threshold = 0.15)
- **Ghost suppression:** tracks confirmed only after 2+ consecutive hits

---

## Tracking Performance

| Frame | Confirmed Tracks | Closest Object | Farthest Object |
|---|---|---|---|
| 000750 | 0 | — (initializing) | — |
| 000751 | 2 | 19.46m | 26.39m |
| 000752 | 5 | 15.23m | 27.61m |
| 000753 | 5 | 8.95m | 27.61m |
| 000754 | **6** | 8.95m | 39.46m |

---

## Dataset

[KITTI Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php)
- 7,481 training frames
- Synchronized LiDAR (Velodyne HDL-64E) + stereo camera + calibration
- Annotations for cars, pedestrians, and cyclists

---

## Local Setup

```bash
git clone https://github.com/gdiaz38/lidar-camera-fusion
cd lidar-camera-fusion
pip install -r requirements.txt

# View results dashboard
streamlit run dashboard.py

# Run full pipeline (requires KITTI dataset)
# Download KITTI and update paths in each notebook
# Run notebooks in order: 01 → 02 → 03 → 04 → 05 → 06
```

---

## Tech Stack

`Python 3.10` · `YOLOv8 (Ultralytics)` · `FilterPy` · `SciPy` · `OpenCV` · `NumPy` · `Streamlit` · `Plotly`

---

## Affiliation

University of California, Riverside — MS in Engineering Management
Part of a portfolio of 10 live data science projects spanning computer vision, NLP, supply chain, and healthcare ML.

---

## License

MIT