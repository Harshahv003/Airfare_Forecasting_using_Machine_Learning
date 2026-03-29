"""
utils/preprocessor.py
----------------------
Mirrors the feature engineering done during model training.
Converts raw user inputs from the web form into the
numeric feature vector expected by the trained model.
"""

import numpy as np
from datetime import datetime


# ── Time-of-day slot helper ───────────────────────────────────────────────────
def get_time_slot(hour: int) -> str:
    if   0  <= hour < 6:  return "Night"
    elif 6  <= hour < 12: return "Morning"
    elif 12 <= hour < 18: return "Afternoon"
    else:                 return "Evening"


# ── Duration string → minutes ─────────────────────────────────────────────────
def duration_to_minutes(dur_str: str) -> int:
    """'2h 50m' → 170, '19h' → 1140, '45m' → 45"""
    dur = str(dur_str).strip()
    hours, minutes = 0, 0
    if "h" in dur:
        parts   = dur.split("h")
        hours   = int(parts[0].strip())
        minutes = int(parts[1].replace("m", "").strip()) if "m" in parts[1] else 0
    elif "m" in dur:
        minutes = int(dur.replace("m", "").strip())
    return hours * 60 + minutes


# ── Main preprocessing function ───────────────────────────────────────────────
def preprocess_input(form_data: dict, encoder_info: dict) -> np.ndarray:
    """
    Convert a dict of raw form values into a 1D numpy array
    matching the model's FEATURES list order.

    Parameters
    ----------
    form_data : {
        "airline":       str,
        "source":        str,
        "destination":   str,
        "journey_date":  str  (YYYY-MM-DD),
        "dep_time":      str  (HH:MM),
        "arrival_time":  str  (HH:MM),
        "duration":      str  (e.g. "2h 30m"),
        "stops":         int  (0,1,2,3,4),
    }

    encoder_info : dict from model bundle — has label classes

    Returns
    -------
    np.ndarray  shape (1, 13)
    """

    # ── Date fields ──────────────────────────────────────────────────────────
    journey_date  = datetime.strptime(form_data["journey_date"], "%Y-%m-%d")
    journey_day   = journey_date.day
    journey_month = journey_date.month

    # ── Departure time ───────────────────────────────────────────────────────
    dep_dt     = datetime.strptime(form_data["dep_time"], "%H:%M")
    dep_hour   = dep_dt.hour
    dep_minute = dep_dt.minute

    # ── Arrival time ─────────────────────────────────────────────────────────
    arr_dt        = datetime.strptime(form_data["arrival_time"], "%H:%M")
    arrival_hour  = arr_dt.hour
    arrival_minute = arr_dt.minute

    # ── Duration ─────────────────────────────────────────────────────────────
    duration_mins = int(form_data.get("duration_minutes", 0))
    if duration_mins == 0:
        # Fallback: compute from dep/arrival
        diff = arr_dt - dep_dt
        duration_mins = int(diff.total_seconds() / 60)
        if duration_mins < 0:                # next-day arrival
            duration_mins += 24 * 60

    # ── Stops ────────────────────────────────────────────────────────────────
    stops = int(form_data.get("stops", 0))

    # ── Time slot ────────────────────────────────────────────────────────────
    time_slot = get_time_slot(dep_hour)

    # ── Route hops (stops + 1 segment minimum) ───────────────────────────────
    route_hops = stops + 1

    # ── Label encode using stored classes ────────────────────────────────────
    def encode(value, classes_list):
        """Map a string value to its integer index. Unknown → 0."""
        try:
            return classes_list.index(value)
        except ValueError:
            return 0

    airline_enc  = encode(form_data["airline"],     encoder_info["airlines"])
    source_enc   = encode(form_data["source"],      encoder_info["sources"])
    dest_enc     = encode(form_data["destination"], encoder_info["destinations"])
    slot_enc     = encode(time_slot,                encoder_info["time_slots"])

    # ── Assemble feature vector (must match FEATURES in train_model.py) ──────
    # FEATURES = [
    #     "Airline_enc", "Source_enc", "Destination_enc",
    #     "Journey_Day", "Journey_Month",
    #     "Dep_Hour", "Dep_Minute",
    #     "Arrival_Hour", "Arrival_Minute",
    #     "Duration_Minutes", "Stops",
    #     "Route_Hops", "TimeSlot_enc",
    # ]
    import pandas as pd

    FEATURES = [
        "Airline_enc", "Source_enc", "Destination_enc",
        "Journey_Day", "Journey_Month",
        "Dep_Hour", "Dep_Minute",
        "Arrival_Hour", "Arrival_Minute",
        "Duration_Minutes", "Stops",
        "Route_Hops", "TimeSlot_enc",
    ]

    features = pd.DataFrame([[
        airline_enc, source_enc, dest_enc,
        journey_day, journey_month,
        dep_hour, dep_minute,
        arrival_hour, arrival_minute,
        duration_mins, stops,
        route_hops, slot_enc,
    ]], columns=FEATURES, dtype=float)

    return features


# ── Input validation ─────────────────────────────────────────────────────────
def validate_input(form_data: dict, encoder_info: dict) -> tuple[bool, str]:
    """
    Validate the user's form data before prediction.

    Returns (is_valid: bool, error_message: str)
    """
    required = ["airline", "source", "destination", "journey_date", "dep_time", "arrival_time"]
    for field in required:
        if not form_data.get(field):
            return False, f"'{field}' is required."

    if form_data["airline"] not in encoder_info["airlines"]:
        return False, f"Unknown airline: {form_data['airline']}"

    if form_data["source"] not in encoder_info["sources"]:
        return False, f"Unknown source city: {form_data['source']}"

    if form_data["destination"] not in encoder_info["destinations"]:
        return False, f"Unknown destination city: {form_data['destination']}"

    if form_data["source"] == form_data["destination"]:
        return False, "Source and destination cannot be the same."

    try:
        datetime.strptime(form_data["journey_date"], "%Y-%m-%d")
    except ValueError:
        return False, "Invalid journey date format. Use YYYY-MM-DD."

    try:
        datetime.strptime(form_data["dep_time"], "%H:%M")
        datetime.strptime(form_data["arrival_time"], "%H:%M")
    except ValueError:
        return False, "Invalid time format. Use HH:MM."

    stops = int(form_data.get("stops", 0))
    if stops < 0 or stops > 4:
        return False, "Stops must be between 0 and 4."

    return True, ""
