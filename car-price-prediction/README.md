# Car Price Predictor (FastAPI + Real CarDekho Dataset)

A FastAPI web app that predicts a used car's resale price using a
RandomForestRegressor trained on the **real CarDekho used-car dataset**
(301 real listings: Car_Name, Year, Selling_Price, Present_Price,
Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner).

## No CSV in this repo, on purpose

`train_model.py` downloads the dataset directly from a public GitHub raw
URL at **build/train time**, trains the model, and discards the CSV --
only the trained model files (`model/car_price_model.pkl`,
`model/encoders.pkl`) are produced, and those aren't committed either
(they're generated fresh on every Render build). This keeps the repo
tiny and avoids bundling any dataset file.

Dataset source (downloaded at build time):
```
https://raw.githubusercontent.com/sumit0072/Car-Price-Prediction-Project/main/car%20data.csv
```
This is the well-known CarDekho used-car dataset (301 listings) commonly
used in used-car price prediction tutorials.

## Project structure

```
car-price-app-fastapi-real/
├── main.py                # FastAPI app (routes + prediction logic)
├── train_model.py         # Downloads real dataset, trains, saves model
├── requirements.txt
├── Procfile
├── render.yaml
├── runtime.txt             # Pins Python to 3.11.9
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Run locally

```bash
pip install -r requirements.txt
python train_model.py         # downloads real data + trains model
uvicorn main:app --reload     # http://localhost:8000
```

Training prints the dataset size and model metrics, e.g.:
```
Loaded 301 rows.
R2 Score: 0.9582
MAE: 0.6372 lakhs
```

## Deploy on Render

1. Push this folder to a GitHub repo (set Render's **Root Directory** to
   this folder's name if it's nested inside a repo, same as before).
2. New Web Service → connect repo.
3. Build Command: `pip install -r requirements.txt && python train_model.py`
4. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Environment variable: `PYTHON_VERSION` = `3.11.9`
6. Deploy -- Render gives you a live URL to submit.


