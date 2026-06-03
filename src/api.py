"""
api.py
======
Flask REST API for IMDb Sentiment Classification.

Endpoints:
  POST /predict        — single text prediction
  POST /predict/batch  — batch prediction (list of texts)
  GET  /health         — health check
  GET  /model/info     — loaded model metadata

Usage:
    python app/api.py
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "This movie was absolutely brilliant!"}'
"""

import os
import time
import logging
import re
import joblib
import numpy as np
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH   = os.getenv("MODEL_PATH", "models/tfidf_linearsvc_tuned.pkl")
MAX_BATCH    = 100
LABEL_MAP    = {0: "negative", 1: "positive"}
_pipeline    = None
_model_meta  = {}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path: str = MODEL_PATH):
    """Load pipeline from disk. Called once at startup."""
    global _pipeline, _model_meta
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Run src/train.py first to generate a trained pipeline."
        )
    logger.info("Loading model from %s ...", path)
    _pipeline = joblib.load(path)
    _model_meta = {
        "model_path":  path,
        "model_name":  os.path.basename(path).replace(".pkl", ""),
        "loaded_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    logger.info("Model loaded: %s", _model_meta["model_name"])


# ─────────────────────────────────────────────────────────────────────────────
# Text preprocessing (mirrors Phase 1 clean_text)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Minimal cleaning applied at inference time."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_single(text: str) -> dict:
    """Run inference on one text string."""
    cleaned  = preprocess(text)
    t0       = time.perf_counter()
    label_id = int(_pipeline.predict([cleaned])[0])
    ms       = round((time.perf_counter() - t0) * 1000, 3)

    confidence = None
    if hasattr(_pipeline, "predict_proba"):
        proba      = _pipeline.predict_proba([cleaned])[0]
        confidence = round(float(max(proba)), 4)

    return {
        "label":         LABEL_MAP[label_id],
        "label_id":      label_id,
        "confidence":    confidence,
        "inference_ms":  ms,
    }


def predict_batch(texts: list) -> list:
    """Run inference on a list of texts."""
    cleaned   = [preprocess(t) for t in texts]
    t0        = time.perf_counter()
    label_ids = _pipeline.predict(cleaned).tolist()
    ms_total  = round((time.perf_counter() - t0) * 1000, 3)

    probas = None
    if hasattr(_pipeline, "predict_proba"):
        probas = _pipeline.predict_proba(cleaned)

    results = []
    for i, (lid, text) in enumerate(zip(label_ids, texts)):
        confidence = round(float(max(probas[i])), 4) if probas is not None else None
        results.append({
            "text":       text[:100] + "..." if len(text) > 100 else text,
            "label":      LABEL_MAP[int(lid)],
            "label_id":   int(lid),
            "confidence": confidence,
        })

    return results, ms_total


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check — returns 200 if model is loaded."""
    if _pipeline is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    return jsonify({"status": "ok", "model": _model_meta.get("model_name")}), 200


@app.route("/model/info", methods=["GET"])
def model_info():
    """Return metadata about the loaded model."""
    if _pipeline is None:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify(_model_meta), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single text prediction.

    Request body (JSON):
        {"text": "This movie was absolutely brilliant!"}

    Response:
        {
          "text":         "This movie was absolutely brilliant!",
          "label":        "positive",
          "label_id":     1,
          "confidence":   0.9821,
          "inference_ms": 1.23
        }
    """
    if _pipeline is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must contain 'text' field"}), 400

    text = str(data["text"]).strip()
    if not text:
        return jsonify({"error": "'text' field is empty"}), 400

    result = predict_single(text)
    result["text"] = text[:200]
    return jsonify(result), 200


@app.route("/predict/batch", methods=["POST"])
def predict_batch_route():
    """
    Batch prediction.

    Request body (JSON):
        {"texts": ["Great film!", "Terrible movie.", "..."]}

    Response:
        {
          "predictions":        [...],
          "count":              3,
          "total_inference_ms": 5.12
        }
    """
    if _pipeline is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data or "texts" not in data:
        return jsonify({"error": "Request body must contain 'texts' list"}), 400

    texts = data["texts"]
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "'texts' must be a non-empty list"}), 400
    if len(texts) > MAX_BATCH:
        return jsonify({"error": f"Batch size exceeds limit of {MAX_BATCH}"}), 400

    predictions, total_ms = predict_batch(texts)
    return jsonify({
        "predictions":        predictions,
        "count":              len(predictions),
        "total_inference_ms": total_ms,
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model(MODEL_PATH)
    logger.info("Starting Flask API on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
