# Sadak Suraksha AI

**From craters to clarity — mapping India's road damage with AI**

An end-to-end road damage detection system for Indian road infrastructure. Upload a road photo or video (or capture one live via webcam) and get back damage classification, severity tagging, and an estimated repair cost.

## What it does

- Detects four categories of road damage — longitudinal cracks, transverse cracks, alligator cracks, and potholes — using a YOLO11 model fine-tuned on real Indian road imagery
- Classifies each detection as **Minor** or **Major** based on relative damage area and detection confidence
- Estimates an approximate repair cost per detection using a damage-type cost lookup table
- Supports image upload, video upload, and live webcam capture
- Returns an annotated image/video with bounding boxes, plus a downloadable summary report

## Tech stack

- **Model:** Ultralytics YOLO11 (nano), fine-tuned via transfer learning
- **Backend:** Flask
- **Inference/vision:** OpenCV
- **Training:** PyTorch, CUDA (GPU-accelerated)
- **Frontend:** Vanilla HTML/CSS/JS

## Dataset

Trained on the India subset of **RDD2022** (Road Damage Dataset), released via CRDDC'2022 — approximately 7,700 training images with PASCAL VOC XML annotations, converted to YOLO format.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
```

Place your trained model weights as `best.pt` in the project root, then run:

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Known limitations

Documented honestly rather than glossed over:

- **Detection recall is approximately 47%** on the current nano model — roughly half of real damage instances may go undetected, particularly in low-light, motion-blurred, or unusually angled photos. A larger model variant (`yolo11s`/`m`) or additional training data would improve this.
- **Cost estimation is illustrative, not verified** — it uses relative bounding-box area as a proxy for physical damage extent, since no camera calibration or GPS-based scale reference is currently used. Treat estimated costs as a rough proof-of-concept, not a quote.
- **Video mode aggregates per-frame, not per unique defect** — the same pothole appearing across multiple consecutive frames is currently counted (and costed) once per frame rather than being tracked as a single object. Object tracking would resolve this.
- **Trained on India data only** — the RDD2022 dataset includes five other countries; cross-region generalization has not been tested and is not expected to be strong without additional training data from those regions.

## Future scope

- Extend training to the Japan subset of RDD2022 (same vehicle-mounted-smartphone capture methodology as India) to improve generalization
- GPS-based geotagging and deduplication of repeated detections
- Object tracking across video frames for accurate per-defect cost aggregation
- Camera calibration for genuine area-to-cost conversion accuracy