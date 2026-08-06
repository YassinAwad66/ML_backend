from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import joblib
import numpy as np

app = FastAPI()

# Allow the React dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model (adjust path/name to whatever you saved)
model = joblib.load("best_xgboost.pkl")
scaler = joblib.load("scaler.pkl")

# Code -> fault mapping, in the SAME order as the frontend expects:
# 0 = Healthy, 1 = TWF, 2 = HDF, 3 = PWF, 4 = OSF, 5 = RNF
FAULT_CODES = {0: "OK", 1: "Tool wear failure", 2: "Heat dissipation failure", 3: "Power failure", 4: "Overstrain failure", 5: "Random failure"}


class MachineInput(BaseModel):
    type: int          # 0=Low, 1=Medium, 2=High
    air_temp: float
    process_temp: float
    speed: float
    torque: float
    tool_wear: float


@app.post("/predict")
def predict(data: MachineInput):
    features = np.array([[
        data.type,
        data.air_temp,
        data.process_temp,
        data.speed,
        data.torque,
        data.tool_wear,
    ]])

    # prediction: 0 = healthy, 1-5 = which fault (see FAULT_CODES above)
    sample_scaled = scaler.transform(features)
    prediction = int(model.predict(sample_scaled)[0])

    # probability of failure, as a 0-100 percentage
    # if your model is binary (healthy vs failure), use predict_proba()[:, 1]
    proba = model.predict_proba(sample_scaled)[0]
    failure_probability = float(np.max(proba[1:]) * 100) if len(proba) > 1 else float(proba[0] * 100)

    return {
        "prediction": prediction,       # <-- the ONLY thing the frontend needs to know the state
        "probability": round(failure_probability, 2),
    }