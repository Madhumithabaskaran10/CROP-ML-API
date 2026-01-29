# =========================================
# PHASE 3.5 – PRICE PREDICTION & SELL ADVICE
# =========================================

print("💰 Phase 3.5 Price & Sell Module Running 💰")

import pandas as pd
import numpy as np
from datetime import datetime

# ===============================
# LOAD DATA
# ===============================
price_df   = pd.read_csv("data/mandi_price_data.csv")
season_df  = pd.read_csv("data/seasonal_price_trend.csv")
storage_df = pd.read_csv("data/crop_storage_data.csv")

# ===============================
# HELPER FUNCTIONS
# ===============================

def get_current_price(crop):
    crop_data = price_df[price_df["crop_name"].str.lower() == crop.lower()]
    return crop_data["price_per_quintal"].mean()


def get_seasonal_multiplier(crop, month):
    crop_data = season_df[season_df["crop_name"].str.lower() == crop.lower()]

    if crop_data.empty:
        return 1.0

    row = crop_data.iloc[0]
    return row.get(month, 1.0)


def predict_future_prices(crop, months_ahead=6):
    current_price = get_current_price(crop)
    predictions = {}

    current_month = datetime.now().month

    for i in range(1, months_ahead + 1):
        future_month = (current_month + i - 1) % 12 + 1
        month_name = datetime(2024, future_month, 1).strftime("%b")

        multiplier = get_seasonal_multiplier(crop, month_name)
        predicted_price = current_price * multiplier

        predictions[f"{i}_month"] = round(predicted_price, 2)

    return predictions


def best_sell_decision(crop):
    current_price = get_current_price(crop)
    future_prices = predict_future_prices(crop)

    best_month = max(future_prices, key=future_prices.get)
    best_price = future_prices[best_month]

    if best_price > current_price * 1.05:
        decision = f"Hold for {best_month.replace('_month','')} month(s)"
        reason = "Expected price increase due to seasonal demand"
    else:
        decision = "Sell Now"
        reason = "Price is stable or storage cost risk is high"

    return {
        "current_price": round(current_price, 2),
        "future_prices": future_prices,
        "best_sell_time": decision,
        "reason": reason
    }

# ===============================
# MAIN PHASE 3.5 FUNCTION
# ===============================

def phase3_5_price_analysis(crop):
    return {
        "crop": crop,
        **best_sell_decision(crop)
    }

# ===============================
# TEST (MANDATORY)
# ===============================
if __name__ == "__main__":

    test_crop = "mungbean"

    result = phase3_5_price_analysis(test_crop)

    print("\n📊 Phase 3.5 Result")
    for k, v in result.items():
        print(f"{k}: {v}")
