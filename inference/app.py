# inference/app.py
import time
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

large_model = joblib.load("models/large_model.joblib")

app = FastAPI()

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)

    start = time.time()
    prediction = large_model.predict(features)[0]
    latency_ms = (time.time() - start) * 1000

    return {
        "prediction": int(prediction),
        "latency_ms": round(latency_ms, 2),
        "model_used": "large"
    }
