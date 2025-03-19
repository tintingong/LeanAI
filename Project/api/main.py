from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np
from pathlib import Path
from os import getenv

app = FastAPI()
templates = Jinja2Templates(directory="api/templates")

MODEL_PATH = Path(getenv("MODEL_PATH", "/app/models/model.pkl"))
print("MODEL_PATH:", MODEL_PATH)

if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        print("---------------   Model loaded")
        print(model)
        print('MODEL TYPE: ', type(model))
else:
    model = None
    print("Model not found or failed to load.")

@app.get("/")
def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict/")
def predict(
    request: Request,
    abdomen: float = Form(...),
    hip: float = Form(...),
    weight: float = Form(...),
    thigh: float = Form(...),
    knee: float = Form(...),
    biceps: float = Form(...),
    neck: float = Form(...),
):
    if model is None:
        return {"error": "Model not found"}

    # Создаем массив с фичами
    features = np.array([[abdomen, weight, chest, hip, thigh, knee, biceps, density]])

    # Получаем предсказание
    print("Input shape:", features.shape)
    prediction = model.predict(features)
    print("Prediction result:", prediction)

    return templates.TemplateResponse("form.html", {
        "request": request,
        "prediction": prediction[0]
    })
