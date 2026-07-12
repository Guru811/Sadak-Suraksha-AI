# Import Libraries
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import os
import uuid

# App Configuration
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO Model
model = YOLO("best.pt")  # Ensure best.pt is in root folder


# Helper Function → Count Classes
# -----------------------------
# Cost estimation table (Rs. per sq. meter of damage, approximate)
# -----------------------------
COST_TABLE = {
    "D00": 150,   # Longitudinal Crack - sealing/patch repair
    "D10": 150,   # Transverse Crack
    "D20": 400,   # Alligator Crack - more extensive resurfacing needed
    "D40": 900,   # Pothole - full-depth patch repair
    "OTHER": 200
}

MAJOR_AREA_THRESHOLD = 0.03
MAJOR_CONF_THRESHOLD = 0.6


def estimate_severity(box_area_frac, confidence):
    if box_area_frac >= MAJOR_AREA_THRESHOLD or confidence >= MAJOR_CONF_THRESHOLD:
        return "Major"
    return "Minor"


def count_detections(results):

    counts = {
        "D00": 0,
        "D10": 0,
        "D20": 0,
        "D40": 0,
        "OTHER": 0
    }

    severity_counts = {"Minor": 0, "Major": 0}
    total_cost = 0
    detections_detail = []

    class_map = {
        "Longitudinal Crack": "D00",
        "Transverse Crack": "D10",
        "Alligator Crack": "D20",
        "Potholes": "D40"
    }

    for r in results:
        if r.boxes is None:
            continue

        img_h, img_w = r.orig_shape
        frame_area = img_h * img_w

        for box in r.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            name = model.names[cls_id]
            code = class_map.get(name, "OTHER")
            counts[code] += 1

            xyxy = box.xyxy[0].tolist()
            box_w = xyxy[2] - xyxy[0]
            box_h = xyxy[3] - xyxy[1]
            box_area_frac = (box_w * box_h) / frame_area if frame_area else 0

            severity = estimate_severity(box_area_frac, conf)
            severity_counts[severity] += 1

            est_area_m2 = round(box_area_frac * 10, 2)
            cost = round(COST_TABLE.get(code, COST_TABLE["OTHER"]) * max(est_area_m2, 0.1))
            total_cost += cost

            detections_detail.append({
                "type": code,
                "severity": severity,
                "confidence": round(conf, 2),
                "estimated_cost": cost
            })

    return counts, severity_counts, total_cost, detections_detail

# Routes
@app.route("/")
def index():
    return render_template("index.html")


# IMAGE PREDICTION
@app.route("/predict_image", methods=["POST"])
def predict_image():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"})

    # Unique filename (avoids overwrite)
    filename = str(uuid.uuid4()) + ".jpg"
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)

    # Run YOLO detection
    results = model(upload_path)
    counts, severity_counts, total_cost, details = count_detections(results)

    # Annotated image
    annotated = results[0].plot()
    result_path = os.path.join(RESULT_FOLDER, filename)
    cv2.imwrite(result_path, annotated)

    return jsonify({
        "result_image": "/" + result_path,
        "counts": counts,
        "severity": severity_counts,
        "estimated_cost_inr": total_cost,
        "details": details
    })

# ===== Progress Tracker =====
video_progress = {
    "percent": 0
}

# VIDEO PREDICTION
@app.route("/predict_video", methods=["POST"])
def predict_video():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    conf = float(request.form.get("confidence", 0.25))

    filename = str(uuid.uuid4()) + ".mp4"
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)

    cap = cv2.VideoCapture(upload_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)

    result_path = os.path.join(
        RESULT_FOLDER,
        "annotated_" + filename
    )

    out = cv2.VideoWriter(
        result_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (width, height)
    )

    counts = {
        "D00": 0,
        "D10": 0,
        "D20": 0,
        "D40": 0,
        "OTHER": 0
    }
    severity_counts = {"Minor": 0, "Major": 0}
    total_cost = 0

    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf)
        frame_counts, frame_severity, frame_cost, _ = count_detections(results)

        # Aggregate counts
        for k in counts:
            counts[k] += frame_counts[k]

        for s in severity_counts:
            severity_counts[s] += frame_severity[s]

        total_cost += frame_cost

        annotated = results[0].plot()
        out.write(annotated)

    cap.release()
    out.release()

    return jsonify({
        "result_video": "/" + result_path,
        "counts": counts,
        "severity": severity_counts,
        "estimated_cost_inr": total_cost
    })


# ==========================================
# Run App
# ==========================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
