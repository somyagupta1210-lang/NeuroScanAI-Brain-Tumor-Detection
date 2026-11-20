"""Flask REST API for the Brain Tumor Detection system."""
import os
import uuid
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from utils import database as db
from utils.model_loader import is_demo_mode, model_info, predict
from utils.preprocess import preprocess_image
from utils.report import build_report

HERE = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(HERE, "uploads")
SAMPLES = os.path.join(HERE, "samples")
REPORTS = os.path.join(HERE, "reports")
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
THRESHOLD = 0.5

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
CORS(app)
db.init_db()


def _run(data: bytes, original_name: str):

    # Generate a unique filename
    ext = os.path.splitext(original_name)[1].lower()
    stored = f"{uuid.uuid4().hex}{ext}"

    # Save uploaded MRI image
    with open(os.path.join(UPLOADS, stored), "wb") as f:
        f.write(data)

    # Preprocess image and get 4-class prediction
    result = predict(preprocess_image(data))

    predicted_class = result["predicted_class"]
    confidence = float(result["confidence"])
    probabilities = result["probabilities"]

    # Determine tumor status
    if predicted_class == "notumor":
        prediction = "No Tumor Detected"
        tumor_probability = 1 - probabilities["notumor"]
    else:
        prediction = "Tumor Detected"
        tumor_probability = 1 - probabilities["notumor"]

    # Save scan details in database
    return db.add_scan(
        filename=original_name,
        stored_name=stored,
        prediction=prediction,
        tumor_prob=float(tumor_probability),
        confidence=confidence,
        tumor_type=predicted_class,
        probabilities=probabilities
    )

@app.get("/api/health")
def health():
    info = model_info()
    return jsonify(status="online", model_mode=info["mode"], model=info)


@app.post("/api/predict")
def api_predict():
    f = request.files.get("image")
    if not f or not f.filename:
        return jsonify(error="No image uploaded."), 400
    if os.path.splitext(f.filename)[1].lower() not in ALLOWED:
        return jsonify(error="Unsupported file type. Use JPG, PNG, BMP or WEBP."), 400
    try:
        return jsonify(_run(f.read(), f.filename))
    except Exception:
        return jsonify(error="Could not read this image. Try another file."), 422


@app.get("/api/samples")
def samples():
    files = sorted(n for n in os.listdir(SAMPLES) if os.path.splitext(n)[1].lower() in ALLOWED)
    return jsonify([{"name": n, "url": f"/api/samples/{n}"} for n in files[:6]])


@app.get("/api/samples/<name>")
def sample_file(name):
    return send_from_directory(SAMPLES, name)


@app.post("/api/predict-sample/<name>")
def predict_sample(name):
    path = os.path.join(SAMPLES, os.path.basename(name))
    if not os.path.exists(path):
        return jsonify(error="Sample not found."), 404
    with open(path, "rb") as f:
        return jsonify(_run(f.read(), os.path.basename(name)))


@app.get("/api/uploads/<name>")
def upload_file(name):
    return send_from_directory(UPLOADS, name)


@app.get("/api/history")
def history():
    return jsonify(db.list_scans())


@app.get("/api/stats")
def stats():
    return jsonify(db.stats())


@app.get("/api/report/<int:scan_id>")
def report(scan_id):
    scan = db.get_scan(scan_id)
    if not scan:
        return jsonify(error="Scan not found."), 404
    out = os.path.join(REPORTS, f"{scan['scan_id']}.pdf")
    stored = scan["image_url"].rsplit("/", 1)[-1]
    build_report(scan, os.path.join(UPLOADS, stored), model_info(), out)
    return send_file(out, as_attachment=True, download_name=f"{scan['scan_id']}_report.pdf")


if __name__ == "__main__":
    print("DEMO MODE - no trained model found. Run train_model.py first." if is_demo_mode() else "Trained model loaded.")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
