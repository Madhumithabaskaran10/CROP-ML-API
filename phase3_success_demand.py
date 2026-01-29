print("🔥 Phase 3 file is running 🔥")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

# =========================
# LOAD DATASETS
# =========================
yield_df  = pd.read_csv("data/crop_yield_history.csv")
season_df = pd.read_csv("data/crop_season_growth.csv")
demand_df = pd.read_csv("data/crop_market_demand.csv")

# =========================
# CLEAN & STANDARDIZE DATA
# =========================
yield_df["crop_name"] = yield_df["crop_name"].str.lower()
season_df["crop_name"] = season_df["crop_name"].str.lower()
demand_df["crop_name"] = demand_df["crop_name"].str.lower()

# Encode soil type (VERY IMPORTANT FIX)
yield_df["soil_type"] = yield_df["soil_type"].astype("category").cat.codes

# =========================
# FEATURES & TARGET
# =========================
X = yield_df[["rainfall", "temperature", "soil_type"]]
y = yield_df["yield_ton_per_hectare"]

# =========================
# CREATE MODELS FOLDER
# =========================
if not os.path.exists("models"):
    os.makedirs("models")

# =========================
# TRAIN MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "models/success_model.pkl")

print("✅ Phase 3 model trained and saved")

# =========================
# SEASON CHECK
# =========================
def check_season(crop, temperature, rainfall):
    row = season_df[season_df["crop_name"] == crop]
    if row.empty:
        return "Unknown"

    r = row.iloc[0]
    if (
        r["ideal_temp_min"] <= temperature <= r["ideal_temp_max"]
        and r["ideal_rainfall_min"] <= rainfall <= r["ideal_rainfall_max"]
    ):
        return "Suitable"
    return "Not Suitable"

# =========================
# DEMAND CHECK
# =========================
def check_demand(crop):
    row = demand_df[demand_df["crop_name"] == crop]
    if row.empty:
        return "Unknown"

    prod = row["total_production"].mean()
    cons = row["total_consumption"].mean()

    if cons > prod:
        return "High"
    elif cons == prod:
        return "Medium"
    return "Low"

# =========================
# PHASE 3 PREDICTION
# =========================
def phase3_predict(crop, rainfall, temperature, soil_type):
    crop = crop.lower()

    model = joblib.load("models/success_model.pkl")

    soil_code = yield_df["soil_type"].mean()

    predicted_yield = model.predict(
        [[rainfall, temperature, soil_code]]
    )[0]

    success_rate = min(
        (predicted_yield / yield_df["yield_ton_per_hectare"].max()) * 100,
        100
    )

    return {
        "Crop": crop,
        "Success Rate (%)": round(success_rate, 2),
        "Season Suitability": check_season(crop, temperature, rainfall),
        "Market Demand": check_demand(crop)
    }

# =========================
# TEST RUN (MANDATORY)
# =========================
if __name__ == "__main__":
    result = phase3_predict(
        crop="mungbean",
        rainfall=600,
        temperature=30,
        soil_type="Loamy"
    )

    print("\n🌾 PHASE 3 OUTPUT 🌾")
    for k, v in result.items():
        print(f"{k}: {v}")
