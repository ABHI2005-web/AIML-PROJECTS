from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np
import os

app = FastAPI(title="Car Price Predictor")

# Serve /static files (CSS etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

MODEL_PATH = os.path.join("model", "car_price_model.pkl")
ENCODERS_PATH = os.path.join("model", "encoders.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODERS_PATH, "rb") as f:
    meta = pickle.load(f)

le_fuel = meta["fuel"]
le_seller = meta["seller"]
le_trans = meta["transmission"]
CURRENT_YEAR = meta["current_year"]
FEATURE_COLS = meta["feature_cols"]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "fuel_options": list(le_fuel.classes_),
            "seller_options": list(le_seller.classes_),
            "trans_options": list(le_trans.classes_),
            "prediction": None,
            "form_data": None,
            "error": None,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    year: int = Form(...),
    present_price: float = Form(...),
    kms_driven: int = Form(...),
    fuel_type: str = Form(...),
    seller_type: str = Form(...),
    transmission: str = Form(...),
    owner: int = Form(...),
):
    form_data = {
        "year": year,
        "present_price": present_price,
        "kms_driven": kms_driven,
        "fuel_type": fuel_type,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
    }

    try:
        car_age = CURRENT_YEAR - year

        fuel_enc = le_fuel.transform([fuel_type])[0]
        seller_enc = le_seller.transform([seller_type])[0]
        trans_enc = le_trans.transform([transmission])[0]

        features = np.array([[
            present_price, kms_driven, car_age,
            fuel_enc, seller_enc, trans_enc, owner
        ]])

        pred = model.predict(features)[0]
        pred = round(float(pred), 2)

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "fuel_options": list(le_fuel.classes_),
                "seller_options": list(le_seller.classes_),
                "trans_options": list(le_trans.classes_),
                "prediction": pred,
                "form_data": form_data,
                "error": None,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "fuel_options": list(le_fuel.classes_),
                "seller_options": list(le_seller.classes_),
                "trans_options": list(le_trans.classes_),
                "prediction": None,
                "form_data": form_data,
                "error": str(e),
            },
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
