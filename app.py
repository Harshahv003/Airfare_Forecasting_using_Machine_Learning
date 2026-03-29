"""
app.py
-------
Airfare Price Prediction — Flask API
Routes:
    GET  /            → main UI
    POST /predict     → return predicted price (JSON)
    GET  /api/meta    → model metrics + supported airlines/cities
    GET  /api/trends  → price trend data for charts
    GET  /health      → uptime check for deployment monitors

Run:  python app.py
Prod: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2
"""

import os
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from utils.predictor import (
    predict_price,
    get_price_range_estimate,
    get_model_metrics,
    get_encoder_info,
)

load_dotenv()

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ── Preload model at startup ─────────────────────────────────────────────────
try:
    _metrics      = get_model_metrics()
    _encoder_info = get_encoder_info()
    print(f"[app] Model loaded: {_metrics['name']}  R²={_metrics['r2']}")
except Exception as e:
    print(f"[app] WARNING: Could not preload model — {e}")
    _metrics, _encoder_info = {}, {}


# ── Helper: load trend data from dataset ─────────────────────────────────────
def _load_trend_data():
    base    = os.path.dirname(os.path.abspath(__file__))
    ds_path = os.path.join(base, "dataset", "Data_Train.xlsx")
    df      = pd.read_excel(ds_path)
    df.dropna(subset=["Route", "Total_Stops"], inplace=True)
    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True)
    df["Journey_Month"]   = df["Date_of_Journey"].dt.month
    return df


# ════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Render main prediction UI."""
    encoder = get_encoder_info()
    metrics = get_model_metrics()
    return render_template(
        "index.html",
        airlines     = encoder.get("airlines", []),
        sources      = encoder.get("sources",  []),
        destinations = encoder.get("destinations", []),
        metrics      = metrics,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body (JSON or form):
        airline, source, destination, journey_date,
        dep_time, arrival_time, stops, duration_minutes
    Returns:
        { success, price, price_low, price_high, error, meta }
    """
    # Accept both JSON and form submissions
    if request.is_json:
        data = request.get_json(force=True)
    else:
        data = request.form.to_dict()

    # Basic presence check
    if not data:
        return jsonify({"success": False, "error": "No input data provided."}), 400

    result = get_price_range_estimate(data)

    status_code = 200 if result["success"] else 400
    return jsonify(result), status_code


@app.route("/api/meta", methods=["GET"])
def api_meta():
    """Return model metadata and supported input values."""
    encoder = get_encoder_info()
    metrics = get_model_metrics()
    return jsonify({
        "model":        metrics,
        "airlines":     encoder.get("airlines", []),
        "sources":      encoder.get("sources",  []),
        "destinations": encoder.get("destinations", []),
        "stops":        [0, 1, 2, 3, 4],
    })


@app.route("/api/trends", methods=["GET"])
def api_trends():
    """
    Returns price trend data for the frontend charts.
    Query params:
        type = monthly | airline | route  (default: monthly)
        airline = <name>   (optional filter)
        source  = <city>   (optional filter)
        dest    = <city>   (optional filter)
    """
    trend_type = request.args.get("type", "monthly")
    try:
        df = _load_trend_data()

        # Optional filters
        airline = request.args.get("airline")
        source  = request.args.get("source")
        dest    = request.args.get("destination")
        if airline: df = df[df["Airline"] == airline]
        if source:  df = df[df["Source"]  == source]
        if dest:    df = df[df["Destination"] == dest]

        if df.empty:
            return jsonify({"error": "No data matches the selected filters."}), 404

        month_map = {3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",
                     8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

        if trend_type == "monthly":
            grp = df.groupby("Journey_Month")["Price"].agg(["mean","min","max"]).reset_index()
            return jsonify({
                "labels": [month_map.get(m, str(m)) for m in grp["Journey_Month"]],
                "avg":    grp["mean"].round(0).tolist(),
                "min":    grp["min"].tolist(),
                "max":    grp["max"].tolist(),
            })

        elif trend_type == "airline":
            grp = df.groupby("Airline")["Price"].mean().sort_values(ascending=False).reset_index()
            return jsonify({
                "labels": grp["Airline"].tolist(),
                "avg":    grp["Price"].round(0).tolist(),
            })

        elif trend_type == "route":
            df["Route_Key"] = df["Source"] + " → " + df["Destination"]
            grp = df.groupby("Route_Key")["Price"].mean().sort_values(ascending=False).head(10).reset_index()
            return jsonify({
                "labels": grp["Route_Key"].tolist(),
                "avg":    grp["Price"].round(0).tolist(),
            })

        else:
            return jsonify({"error": f"Unknown trend type: {trend_type}"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Render/Railway deployment."""
    return jsonify({"status": "ok", "model_loaded": bool(_metrics)}), 200


# ── Error handlers ───────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found."}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    port  = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug)
