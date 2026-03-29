"""
utils/predictor.py
-------------------
Loads the trained model bundle and exposes a clean
predict_price() function that the Flask app and any
external script can call.
"""

import os
import pickle
import numpy as np
from utils.preprocessor import preprocess_input, validate_input

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "flight_price_model.pkl")

# ── Lazy-load model once ──────────────────────────────────────────────────────
_bundle = None

def _load_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run: python model/train_model.py"
            )
        with open(MODEL_PATH, "rb") as f:
            _bundle = pickle.load(f)
    return _bundle


def get_model_metrics() -> dict:
    """Return the performance metrics of the loaded model."""
    bundle = _load_bundle()
    return bundle["metrics"]


def get_encoder_info() -> dict:
    """Return the label-encoder metadata (airline names, cities, etc.)."""
    bundle = _load_bundle()
    return bundle["encoder_info"]


def predict_price(form_data: dict) -> dict:
    """
    Predict airfare from raw form data.

    Parameters
    ----------
    form_data : dict with keys:
        airline, source, destination, journey_date,
        dep_time, arrival_time, stops, duration_minutes (optional)

    Returns
    -------
    dict:
        {
            "success": bool,
            "price":   float or None,
            "error":   str or None,
            "meta":    { model_name, r2, mae, rmse }
        }
    """
    try:
        bundle       = _load_bundle()
        model        = bundle["model"]
        encoder_info = bundle["encoder_info"]
        metrics      = bundle["metrics"]

        # Validate
        is_valid, err_msg = validate_input(form_data, encoder_info)
        if not is_valid:
            return {"success": False, "price": None, "error": err_msg, "meta": metrics}

        # Preprocess
        features = preprocess_input(form_data, encoder_info)

        # Predict
        raw_price = float(model.predict(features)[0])
        price     = max(500.0, round(raw_price, 2))   # floor ₹500

        return {
            "success": True,
            "price":   price,
            "error":   None,
            "meta":    metrics,
        }

    except FileNotFoundError as e:
        return {"success": False, "price": None, "error": str(e), "meta": {}}
    except Exception as e:
        return {"success": False, "price": None, "error": f"Prediction failed: {str(e)}", "meta": {}}


def get_price_range_estimate(form_data: dict) -> dict:
    """
    Returns a ±10 % confidence band around the predicted price.
    Useful for displaying a price range to the user.
    """
    result = predict_price(form_data)
    if not result["success"]:
        return result

    price = result["price"]
    result["price_low"]  = round(price * 0.90, 2)
    result["price_high"] = round(price * 1.10, 2)
    return result
