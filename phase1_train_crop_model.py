# ==============================
# PHASE 1: CROP RECOMMENDATION MODEL
# ==============================

print("Phase 1 started...")

# 1. Import libraries
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 2. Load dataset
DATA_PATH = "data/crop_data.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print(df.head())

# 3. Clean dataset
df = df.dropna()

# 4. Encode target column (Crop)
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

# 5. Split features and target
X = df.drop("label", axis=1)
y = df["label"]

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# 8. Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 9. Create models folder
os.makedirs("models", exist_ok=True)

# 10. Save model & encoder
joblib.dump(model, "models/crop_rf_model.pkl")
joblib.dump(le, "models/crop_label_encoder.pkl")

print("Model & Label Encoder saved successfully!")
print("Phase 1 completed ✅")
def get_recommended_crops(n, p, k, ph, rainfall, temperature):
    model = joblib.load("models/crop_model.pkl")

    input_data = [[n, p, k, ph, rainfall, temperature]]
    probabilities = model.predict_proba(input_data)[0]

    classes = model.classes_
    sorted_indices = probabilities.argsort()[::-1]

    primary_crop = classes[sorted_indices[0]]
    backup_crop = classes[sorted_indices[1]]

    return {
        "primary_crop": primary_crop.lower(),
        "backup_crop": backup_crop.lower()
    }


