from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np
from pathlib import Path

app = FastAPI()
templates = Jinja2Templates(directory="api/templates")

# Load model
MODEL_PATH = Path("models/model.pkl")
if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None

@app.get("/")
def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict/")
def predict(request: Request, feature1: float = Form(...), feature2: float = Form(...)):
    if not model:
        return {"error": "Model not found"}

    features = np.array([[feature1, feature2]])
    prediction = model.predict(features)
    return templates.TemplateResponse("form.html", {
        "request": request,
        "prediction": prediction[0]
    })
