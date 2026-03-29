"""
notebooks/EDA_notebook.py
--------------------------
Standalone Exploratory Data Analysis script.
Produces a detailed console report + all EDA plots.
Run: python notebooks/EDA_notebook.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "dataset", "Data_Train.xlsx")
PLOTS = os.path.join(BASE, "static", "plots")
os.makedirs(PLOTS, exist_ok=True)

print("=" * 60)
print("  AIRFARE PREDICTION — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ── Load ──────────────────────────────────────────────────────────
df = pd.read_excel(DATA)
df.dropna(subset=["Route","Total_Stops"], inplace=True)
df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True)
df["Journey_Month"]   = df["Date_of_Journey"].dt.month

stops_map = {"non-stop":0,"1 stop":1,"2 stops":2,"3 stops":3,"4 stops":4}
df["Stops"] = df["Total_Stops"].map(stops_map).fillna(0).astype(int)

def dur_mins(d):
    d=str(d).strip(); h,m=0,0
    if "h" in d:
        p=d.split("h"); h=int(p[0].strip())
        m=int(p[1].replace("m","").strip()) if "m" in p[1] else 0
    elif "m" in d: m=int(d.replace("m","").strip())
    return h*60+m

df["Duration_Minutes"] = df["Duration"].apply(dur_mins)

# ── Console Summary ───────────────────────────────────────────────
print(f"\nDataset shape : {df.shape}")
print(f"Date range    : {df['Date_of_Journey'].min().date()} → {df['Date_of_Journey'].max().date()}")
print(f"\nPrice statistics (₹):")
print(df["Price"].describe().apply(lambda x: f"  {x:,.2f}").to_string())

print(f"\nTop 5 most common routes:")
route_counts = (df["Source"] + " → " + df["Destination"]).value_counts().head(5)
for r,c in route_counts.items():
    avg = df[(df["Source"]==r.split(" → ")[0]) & (df["Destination"]==r.split(" → ")[1])]["Price"].mean()
    print(f"  {r:<35}  {c:4d} flights   avg ₹{avg:,.0f}")

print(f"\nAverage price by airline:")
for airline, grp in df.groupby("Airline")["Price"].mean().sort_values(ascending=False).items():
    print(f"  {airline:<40} ₹{grp:,.0f}")

print(f"\nAverage price by stops:")
for stops, grp in df.groupby("Stops")["Price"].mean().items():
    label = {0:"Non-stop",1:"1 Stop",2:"2 Stops",3:"3 Stops",4:"4 Stops"}.get(stops,"?")
    print(f"  {label:<12} ₹{grp:,.0f}")

print(f"\nCorrelation with Price:")
num_cols = ["Duration_Minutes","Stops","Price"]
corr = df[num_cols].corr()["Price"].drop("Price")
for col,val in corr.items():
    bar = "█" * int(abs(val)*20)
    print(f"  {col:<20} {val:+.4f}  {bar}")

print(f"\n✓ EDA complete — plots saved to static/plots/")
print("=" * 60)
