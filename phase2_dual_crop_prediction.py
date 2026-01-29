# ==============================
# PHASE 2: PRIMARY + BACKUP CROP PREDICTION
# ==============================

import numpy as np
import joblib

# 1. Load trained model and encoder
model = joblib.load("models/crop_rf_model.pkl")
label_encoder = joblib.load("models/crop_label_encoder.pkl")

print("Model and encoder loaded successfully")

# 2. Take input from user
print("\nEnter soil and climate values:")

N = float(input("Nitrogen (N): "))
P = float(input("Phosphorus (P): "))
K = float(input("Potassium (K): "))
temperature = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
ph = float(input("pH value: "))
rainfall = float(input("Rainfall (mm): "))

# 3. Prepare input for model
input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

# 4. Predict probabilities
probabilities = model.predict_proba(input_data)[0]

# 5. Get top 2 crops
top_two_indices = np.argsort(probabilities)[::-1][:2]
top_two_crops = label_encoder.inverse_transform(top_two_indices)
top_two_probs = probabilities[top_two_indices]

# 6. Display results
print("\n🌾 CROP RECOMMENDATION 🌾")
print(f"🥇 Primary Crop : {top_two_crops[0]}  ({top_two_probs[0]*100:.2f}%)")
print(f"🥈 Backup Crop  : {top_two_crops[1]}  ({top_two_probs[1]*100:.2f}%)")
