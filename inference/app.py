import time
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from inference.latency import LatencyMonitor


# Load models once at startup
large_model = joblib.load("models/large_model.joblib")
small_model = joblib.load("models/small_model.joblib")

# Initialize latency monitor
latency_monitor = LatencyMonitor(window_size=50, sla_ms=50)

app = FastAPI()

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)

    start = time.time()
    prediction = large_model.predict(features)[0]
    latency_ms = (time.time() - start) * 1000

    # Record latency
    latency_monitor.record(latency_ms)

    return {
        "prediction": int(prediction),
        "latency_ms": round(latency_ms, 2),
        "avg_latency_ms": round(latency_monitor.avg_latency(), 2),
        "p95_latency_ms": round(latency_monitor.p95_latency(), 2),
        "sla_violated": latency_monitor.sla_violated(),
        "model_used": "large"
    }
