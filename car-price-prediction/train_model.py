"""
train_model.py
Downloads the real CarDekho used-car dataset (Car_Name, Year, Selling_Price,
Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner)
at build/train time -- the CSV itself is never stored in this repo -- then
trains a RandomForestRegressor on it and saves the model + encoders.
"""

import io
import os
import pickle

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Public raw CSV of the well-known 301-row CarDekho used-car dataset
# (Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type,
# Seller_Type, Transmission, Owner). Downloaded fresh at build time so the
# file never has to live inside this repository.
DATASET_URL = (
    "https://raw.githubusercontent.com/sumit0072/"
    "Car-Price-Prediction-Project/main/car%20data.csv"
)

print(f"Downloading dataset from {DATASET_URL} ...")
response = requests.get(DATASET_URL, timeout=30)
response.raise_for_status()

df = pd.read_csv(io.StringIO(response.text))
print(f"Loaded {len(df)} rows.")
print(df.head())

CURRENT_YEAR = 2026
df["Car_Age"] = CURRENT_YEAR - df["Year"]

le_fuel = LabelEncoder()
le_seller = LabelEncoder()
le_trans = LabelEncoder()

df["Fuel_Type_enc"] = le_fuel.fit_transform(df["Fuel_Type"])
df["Seller_Type_enc"] = le_seller.fit_transform(df["Seller_Type"])
df["Transmission_enc"] = le_trans.fit_transform(df["Transmission"])

feature_cols = [
    "Present_Price", "Kms_Driven", "Car_Age",
    "Fuel_Type_enc", "Seller_Type_enc", "Transmission_enc", "Owner"
]

X = df[feature_cols]
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f} lakhs")

os.makedirs("model", exist_ok=True)

with open("model/car_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/encoders.pkl", "wb") as f:
    pickle.dump({
        "fuel": le_fuel,
        "seller": le_seller,
        "transmission": le_trans,
        "current_year": CURRENT_YEAR,
        "feature_cols": feature_cols
    }, f)

print("Model and encoders saved to /model")
