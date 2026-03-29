"""
model/train_model.py
---------------------
Complete ML pipeline for Airfare Price Prediction.
Trains Linear Regression, Random Forest, and Gradient Boosting.
Selects the best model and saves it via pickle.

Run:  python model/train_model.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "Data_Train.xlsx")
MODEL_PATH   = os.path.join(BASE_DIR, "model",   "flight_price_model.pkl")
PLOT_DIR     = os.path.join(BASE_DIR, "static",  "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 60)
print("  AIRFARE PRICE PREDICTION — ML TRAINING PIPELINE")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════
print("\n[1/7] Loading dataset...")
df = pd.read_excel(DATASET_PATH)
print(f"      Shape: {df.shape}  |  Columns: {df.columns.tolist()}")

# ═══════════════════════════════════════════════════════════════
# STEP 2 — CLEAN DATA
# ═══════════════════════════════════════════════════════════════
print("\n[2/7] Cleaning data...")

# Drop rows with missing Route / Total_Stops (only 1-2 rows)
df.dropna(subset=["Route", "Total_Stops"], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"      After dropping nulls: {df.shape}")

# ═══════════════════════════════════════════════════════════════
# STEP 3 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
print("\n[3/7] Engineering features...")

# --- Date of Journey ---
df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True)
df["Journey_Day"]     = df["Date_of_Journey"].dt.day
df["Journey_Month"]   = df["Date_of_Journey"].dt.month

# --- Departure time ---
df["Dep_Hour"]   = pd.to_datetime(df["Dep_Time"], format="%H:%M").dt.hour
df["Dep_Minute"] = pd.to_datetime(df["Dep_Time"], format="%H:%M").dt.minute

# --- Arrival time (handle "01:10 22 Mar" style) ---
def parse_arrival_hour(s):
    try:
        return int(str(s).strip().split(":")[0])
    except Exception:
        return 0

def parse_arrival_minute(s):
    try:
        part = str(s).strip().split(":")[1]
        return int(part[:2])
    except Exception:
        return 0

df["Arrival_Hour"]   = df["Arrival_Time"].apply(parse_arrival_hour)
df["Arrival_Minute"] = df["Arrival_Time"].apply(parse_arrival_minute)

# --- Duration → total minutes ---
def duration_to_minutes(dur):
    """Convert '2h 50m', '7h 25m', '19h', '45m' → integer minutes."""
    dur = str(dur).strip()
    hours, minutes = 0, 0
    if "h" in dur:
        parts = dur.split("h")
        hours = int(parts[0].strip())
        if "m" in parts[1]:
            minutes = int(parts[1].replace("m", "").strip())
    elif "m" in dur:
        minutes = int(dur.replace("m", "").strip())
    return hours * 60 + minutes

df["Duration_Minutes"] = df["Duration"].apply(duration_to_minutes)

# --- Total Stops → integer ---
stops_map = {
    "non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4
}
df["Stops"] = df["Total_Stops"].map(stops_map).fillna(0).astype(int)

# --- Time-of-day slot for departure ---
def time_slot(hour):
    if   0  <= hour < 6:  return "Night"
    elif 6  <= hour < 12: return "Morning"
    elif 12 <= hour < 18: return "Afternoon"
    else:                 return "Evening"

df["Dep_TimeSlot"] = df["Dep_Hour"].apply(time_slot)

# --- Number of hops from Route ---
df["Route_Hops"] = df["Route"].apply(lambda r: len(str(r).split("→")))

# --- Label encode categoricals ---
le_airline = LabelEncoder()
le_source  = LabelEncoder()
le_dest    = LabelEncoder()
le_slot    = LabelEncoder()

df["Airline_enc"]     = le_airline.fit_transform(df["Airline"])
df["Source_enc"]      = le_source.fit_transform(df["Source"])
df["Destination_enc"] = le_dest.fit_transform(df["Destination"])
df["TimeSlot_enc"]    = le_slot.fit_transform(df["Dep_TimeSlot"])

# ─── Save encoder classes for use in prediction ───────────────
encoder_info = {
    "airlines":      list(le_airline.classes_),
    "sources":       list(le_source.classes_),
    "destinations":  list(le_dest.classes_),
    "time_slots":    list(le_slot.classes_),
    "stops_map":     stops_map,
}

print("      Airlines:     ", list(le_airline.classes_))
print("      Sources:      ", list(le_source.classes_))
print("      Destinations: ", list(le_dest.classes_))

# ═══════════════════════════════════════════════════════════════
# STEP 4 — EDA PLOTS (saved to static/plots/)
# ═══════════════════════════════════════════════════════════════
print("\n[4/7] Generating EDA plots...")

plt.style.use("seaborn-v0_8-darkgrid")
COLORS = ["#4361ee", "#3a0ca3", "#7209b7", "#f72585", "#4cc9f0"]

# Plot 1 — Price distribution
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df["Price"], bins=60, color="#4361ee", alpha=0.85, edgecolor="white")
ax.set_title("Flight Price Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Price (₹)")
ax.set_ylabel("Count")
ax.axvline(df["Price"].median(), color="#f72585", linestyle="--", label=f'Median ₹{df["Price"].median():,.0f}')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "price_dist.png"), dpi=120)
plt.close()

# Plot 2 — Airline vs Avg Price
fig, ax = plt.subplots(figsize=(12, 5))
airline_avg = df.groupby("Airline")["Price"].mean().sort_values(ascending=False)
bars = ax.bar(airline_avg.index, airline_avg.values, color=COLORS * 4)
ax.set_title("Average Price by Airline", fontsize=14, fontweight="bold")
ax.set_xlabel("Airline")
ax.set_ylabel("Avg Price (₹)")
ax.set_xticklabels(airline_avg.index, rotation=35, ha="right")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'₹{bar.get_height():,.0f}', ha="center", va="bottom", fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "airline_price.png"), dpi=120)
plt.close()

# Plot 3 — Stops vs Price
fig, ax = plt.subplots(figsize=(8, 5))
df.groupby("Stops")["Price"].mean().plot(kind="bar", ax=ax, color=COLORS[:5], edgecolor="white")
ax.set_title("Average Price by Number of Stops", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Stops")
ax.set_ylabel("Avg Price (₹)")
ax.set_xticklabels(["Non-stop", "1 Stop", "2 Stops", "3 Stops", "4 Stops"], rotation=0)
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "stops_price.png"), dpi=120)
plt.close()

# Plot 4 — Duration vs Price scatter
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(df["Duration_Minutes"], df["Price"], alpha=0.25, color="#7209b7", s=10)
ax.set_title("Duration vs Price", fontsize=14, fontweight="bold")
ax.set_xlabel("Duration (minutes)")
ax.set_ylabel("Price (₹)")
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "duration_price.png"), dpi=120)
plt.close()

# Plot 5 — Source vs Destination heatmap (avg price)
fig, ax = plt.subplots(figsize=(10, 5))
pivot = df.pivot_table(values="Price", index="Source", columns="Destination", aggfunc="mean")
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, linewidths=0.5)
ax.set_title("Average Price Heatmap: Source → Destination", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "route_heatmap.png"), dpi=120)
plt.close()

# Plot 6 — Month vs avg price
fig, ax = plt.subplots(figsize=(10, 4))
month_names = {3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
monthly = df.groupby("Journey_Month")["Price"].mean()
ax.plot(monthly.index, monthly.values, marker="o", color="#f72585", linewidth=2)
ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="#f72585")
ax.set_xticks(monthly.index)
ax.set_xticklabels([month_names.get(m, m) for m in monthly.index])
ax.set_title("Average Price by Month", fontsize=14, fontweight="bold")
ax.set_ylabel("Avg Price (₹)")
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "monthly_price.png"), dpi=120)
plt.close()

print("      Saved 6 EDA plots to static/plots/")

# ═══════════════════════════════════════════════════════════════
# STEP 5 — PREPARE FEATURES
# ═══════════════════════════════════════════════════════════════
print("\n[5/7] Preparing feature matrix...")

FEATURES = [
    "Airline_enc", "Source_enc", "Destination_enc",
    "Journey_Day", "Journey_Month",
    "Dep_Hour", "Dep_Minute",
    "Arrival_Hour", "Arrival_Minute",
    "Duration_Minutes", "Stops",
    "Route_Hops", "TimeSlot_enc",
]

X = df[FEATURES]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"      Train: {X_train.shape}  |  Test: {X_test.shape}")

# ═══════════════════════════════════════════════════════════════
# STEP 6 — TRAIN & COMPARE MODELS
# ═══════════════════════════════════════════════════════════════
print("\n[6/7] Training models...")

def evaluate(name, model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    r2   = r2_score(yte, pred)
    mae  = mean_absolute_error(yte, pred)
    rmse = np.sqrt(mean_squared_error(yte, pred))
    print(f"      {name:<35}  R²={r2:.4f}  MAE=₹{mae:,.0f}  RMSE=₹{rmse:,.0f}")
    return {"name": name, "model": model, "r2": r2, "mae": mae, "rmse": rmse, "pred": pred}

results = []

# 1. Linear Regression (baseline)
results.append(evaluate("Linear Regression (baseline)",
    LinearRegression(), X_train, X_test, y_train, y_test))

# 2. Random Forest
results.append(evaluate("Random Forest (default)",
    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    X_train, X_test, y_train, y_test))

# 3. Gradient Boosting (XGBoost-style)
results.append(evaluate("Gradient Boosting (XGBoost-style)",
    GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                              max_depth=5, random_state=42),
    X_train, X_test, y_train, y_test))

# ── Pick best by R² ──────────────────────────────────────────
best_result = max(results, key=lambda r: r["r2"])
print(f"\n      ✓ Best model: {best_result['name']}  (R²={best_result['r2']:.4f})")

# ── Hyperparameter tuning on Random Forest ──────────────────
print("\n      Hyperparameter tuning on Random Forest (RandomizedSearchCV)...")
param_dist = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2"],
}
rf_tuned = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=20, cv=3, scoring="r2", random_state=42, n_jobs=-1
)
rf_tuned.fit(X_train, y_train)
pred_tuned = rf_tuned.best_estimator_.predict(X_test)
r2_tuned   = r2_score(y_test, pred_tuned)
mae_tuned  = mean_absolute_error(y_test, pred_tuned)
rmse_tuned = np.sqrt(mean_squared_error(y_test, pred_tuned))
print(f"      Tuned RF — R²={r2_tuned:.4f}  MAE=₹{mae_tuned:,.0f}  RMSE=₹{rmse_tuned:,.0f}")
print(f"      Best params: {rf_tuned.best_params_}")

# Use tuned RF if better than all defaults
if r2_tuned > best_result["r2"]:
    final_model = rf_tuned.best_estimator_
    final_name  = "Random Forest (Tuned)"
    final_r2    = r2_tuned
    final_mae   = mae_tuned
    final_rmse  = rmse_tuned
    final_pred  = pred_tuned
else:
    final_model = best_result["model"]
    final_name  = best_result["name"]
    final_r2    = best_result["r2"]
    final_mae   = best_result["mae"]
    final_rmse  = best_result["rmse"]
    final_pred  = best_result["pred"]

print(f"\n      ✓ Final model selected: {final_name}")

# ── Comparison bar chart ─────────────────────────────────────
all_names = [r["name"].replace(" (baseline)","").replace(" (default)","") for r in results]
all_names.append("Random Forest (Tuned)")
all_r2 = [r["r2"] for r in results] + [r2_tuned]

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.barh(all_names, all_r2, color=["#adb5bd","#4361ee","#7209b7","#f72585"])
ax.set_xlim(0, 1.05)
ax.set_title("Model Comparison — R² Score", fontsize=14, fontweight="bold")
ax.set_xlabel("R² Score (higher = better)")
for bar in bars:
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{bar.get_width():.4f}', va="center", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "model_comparison.png"), dpi=120)
plt.close()

# ── Actual vs Predicted ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, final_pred, alpha=0.25, color="#4361ee", s=10)
lims = [min(y_test.min(), final_pred.min()), max(y_test.max(), final_pred.max())]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Price (₹)")
ax.set_ylabel("Predicted Price (₹)")
ax.set_title(f"Actual vs Predicted — {final_name}", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "actual_vs_predicted.png"), dpi=120)
plt.close()

# ── Feature importance ────────────────────────────────────────
if hasattr(final_model, "feature_importances_"):
    feat_imp = pd.Series(final_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    feat_imp.plot(kind="barh", ax=ax, color="#4361ee")
    ax.set_title("Feature Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "feature_importance.png"), dpi=120)
    plt.close()

print("      Saved model evaluation plots.")

# ═══════════════════════════════════════════════════════════════
# STEP 7 — SAVE MODEL + META
# ═══════════════════════════════════════════════════════════════
print("\n[7/7] Saving model...")

model_bundle = {
    "model":         final_model,
    "features":      FEATURES,
    "encoder_info":  encoder_info,
    "metrics": {
        "name": final_name,
        "r2":   round(final_r2,  4),
        "mae":  round(final_mae, 2),
        "rmse": round(final_rmse, 2),
    },
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_bundle, f)

print(f"      Model saved → {MODEL_PATH}")
print(f"\n{'='*60}")
print(f"  TRAINING COMPLETE")
print(f"  Model : {final_name}")
print(f"  R²    : {final_r2:.4f}")
print(f"  MAE   : ₹{final_mae:,.2f}")
print(f"  RMSE  : ₹{final_rmse:,.2f}")
print(f"{'='*60}\n")
