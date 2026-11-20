"""Flask REST API for the Brain Tumor Detection system."""

import os
import uuid
import traceback

from flask import (
    Flask,
    jsonify,
    request,
    send_file,
    send_from_directory
)

from flask_cors import CORS

from utils import database as db

from utils.model_loader import (
    is_demo_mode,
    model_info,
    predict,
    generate_gradcam,
    CLASS_NAMES
)

from utils.gradcam_visualization import create_gradcam_overlay
from utils.preprocess import preprocess_image
from utils.report import build_report


# ==============================
# PROJECT DIRECTORIES
# ==============================

HERE = os.path.dirname(os.path.abspath(__file__))

UPLOADS = os.path.join(HERE, "uploads")
SAMPLES = os.path.join(HERE, "samples")
REPORTS = os.path.join(HERE, "reports")
GRADCAM_DIR = os.path.join(HERE, "gradcam")

for folder in [UPLOADS, REPORTS, GRADCAM_DIR]:
    os.makedirs(folder, exist_ok=True)


ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

THRESHOLD = 0.5


# ==============================
# FLASK CONFIGURATION
# ==============================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

CORS(app)

db.init_db()


# ==============================
# MAIN PREDICTION FUNCTION
# ==============================

def _run(data: bytes, original_name: str):

    print("\n========== NEW MRI SCAN ==========")

    ext = os.path.splitext(original_name)[1].lower()

    if ext not in ALLOWED:
        raise ValueError("Unsupported image format.")

    stored = f"{uuid.uuid4().hex}{ext}"

    image_path = os.path.join(UPLOADS, stored)

    # Save uploaded MRI image
    with open(image_path, "wb") as f:
        f.write(data)

    print("DEBUG 1: Image saved:", image_path)

    # Preprocess image
    batch = preprocess_image(data)

    print("DEBUG 2: Image preprocessing successful")

    # CNN Prediction
    result = predict(batch)

    print("DEBUG 3: CNN prediction successful")

    predicted_class = result["predicted_class"]

    confidence = float(result["confidence"])

    probabilities = result["probabilities"]

    tumor_probability = 1 - float(probabilities["notumor"])

    if predicted_class == "notumor":
        prediction = "No Tumor Detected"
    else:
        prediction = "Tumor Detected"

    print("DEBUG 4: Predicted class:", predicted_class)

    # ==============================
    # GRAD-CAM GENERATION
    # ==============================

    gradcam_url = None
    gradcam_error = None

    try:

        print("DEBUG 5: Starting Grad-CAM")

        class_index = CLASS_NAMES.index(predicted_class)

        print("DEBUG 6: Class index:", class_index)

        heatmap = generate_gradcam(batch, class_index)

        print(
            "DEBUG 7: Heatmap generated:",
            heatmap is not None
        )

        if heatmap is None:
            raise ValueError("Grad-CAM returned an empty heatmap.")

        overlay_name = (
            f"gradcam_{os.path.splitext(stored)[0]}.jpg"
        )

        overlay_path = os.path.join(
            GRADCAM_DIR,
            overlay_name
        )

        create_gradcam_overlay(
            data,
            heatmap,
            overlay_path
        )

        if not os.path.exists(overlay_path):
            raise FileNotFoundError(
                "Grad-CAM overlay was not saved."
            )

        gradcam_url = f"/api/gradcam/{overlay_name}"

        print("DEBUG 8: Grad-CAM overlay created:", overlay_path)

    except Exception as e:

        gradcam_error = str(e)

        print("GRADCAM ERROR:", gradcam_error)

        traceback.print_exc()

    # ==============================
    # SAVE SCAN IN DATABASE
    # ==============================

    scan = db.add_scan(
        filename=original_name,
        stored_name=stored,
        prediction=prediction,
        tumor_prob=float(tumor_probability),
        confidence=confidence,
        tumor_type=predicted_class,
        probabilities=probabilities
    )

    scan["gradcam_url"] = gradcam_url

    if gradcam_error:
        scan["gradcam_error"] = gradcam_error

    print("DEBUG 9: Scan saved successfully")

    print("DEBUG GRADCAM:", gradcam_url)

    print("========== SCAN COMPLETED ==========\n")

    return scan


# ==============================
# HEALTH CHECK API
# ==============================

@app.get("/api/health")
def health():

    info = model_info()

    return jsonify(
        status="online",
        model_mode=info["mode"],
        model=info
    )


# ==============================
# PREDICT UPLOADED MRI
# ==============================

@app.post("/api/predict")
def api_predict():

    f = request.files.get("image")

    if not f or not f.filename:
        return jsonify(error="No image uploaded."), 400

    if os.path.splitext(f.filename)[1].lower() not in ALLOWED:
        return jsonify(
            error="Unsupported file type."
        ), 400

    try:

        result = _run(f.read(), f.filename)

        return jsonify(result)

    except Exception as e:

        print("PREDICTION ERROR:", str(e))

        traceback.print_exc()

        return jsonify(
            error=str(e)
        ), 422


# ==============================
# GET SAMPLE IMAGES
# ==============================

@app.get("/api/samples")
def samples():

    files = sorted(
        n for n in os.listdir(SAMPLES)
        if os.path.splitext(n)[1].lower() in ALLOWED
    )

    return jsonify([
        {
            "name": n,
            "url": f"/api/samples/{n}"
        }
        for n in files[:6]
    ])


# ==============================
# SERVE SAMPLE IMAGE
# ==============================

@app.get("/api/samples/<name>")
def sample_file(name):

    return send_from_directory(SAMPLES, name)


# ==============================
# PREDICT SAMPLE IMAGE
# ==============================

@app.post("/api/predict-sample/<name>")
def predict_sample(name):

    safe_name = os.path.basename(name)

    path = os.path.join(SAMPLES, safe_name)

    if not os.path.exists(path):
        return jsonify(error="Sample not found."), 404

    try:

        with open(path, "rb") as f:
            data = f.read()

        result = _run(data, safe_name)

        return jsonify(result)

    except Exception as e:

        print("SAMPLE PREDICTION ERROR:", str(e))

        traceback.print_exc()

        return jsonify(error=str(e)), 422


# ==============================
# SERVE UPLOADED IMAGE
# ==============================

@app.get("/api/uploads/<name>")
def upload_file(name):

    return send_from_directory(UPLOADS, name)


# ==============================
# SERVE GRAD-CAM IMAGE
# ==============================

@app.get("/api/gradcam/<name>")
def gradcam_file(name):

    return send_from_directory(GRADCAM_DIR, name)


# ==============================
# SCAN HISTORY
# ==============================

@app.get("/api/history")
def history():

    scans = db.list_scans()

    for scan in scans:

        stored_name = scan["image_url"].rsplit("/", 1)[-1]

        overlay_name = (
            f"gradcam_{os.path.splitext(stored_name)[0]}.jpg"
        )

        overlay_path = os.path.join(
            GRADCAM_DIR,
            overlay_name
        )

        if os.path.exists(overlay_path):
            scan["gradcam_url"] = f"/api/gradcam/{overlay_name}"
        else:
            scan["gradcam_url"] = None

    print("DEBUG: History loaded successfully")

    return jsonify(scans)


# ==============================
# STATISTICS
# ==============================

@app.get("/api/stats")
def stats():

    return jsonify(db.stats())


# ==============================
# GENERATE PDF REPORT
# ==============================

@app.get("/api/report/<int:scan_id>")
def report(scan_id):

    scan = db.get_scan(scan_id)

    if not scan:
        return jsonify(error="Scan not found."), 404

    out = os.path.join(
        REPORTS,
        f"{scan['scan_id']}.pdf"
    )

    stored = scan["image_url"].rsplit("/", 1)[-1]

    build_report(
        scan,
        os.path.join(UPLOADS, stored),
        model_info(),
        out
    )

    return send_file(
        out,
        as_attachment=True,
        download_name=f"{scan['scan_id']}_report.pdf"
    )


# ==============================
# START FLASK SERVER
# ==============================

if __name__ == "__main__":

    print("\n================================")

    print(
        "DEMO MODE - no trained model found."
        if is_demo_mode()
        else "Trained CNN model loaded successfully."
    )

    print("Grad-CAM debugging enabled.")

    print("================================\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )