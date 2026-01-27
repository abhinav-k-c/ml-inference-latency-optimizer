import time
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from inference.latency import LatencyMonitor
from inference.router import ModelRouter

# Load models
large_model = joblib.load("models/large_model.joblib")
small_model = joblib.load("models/small_model.joblib")

latency_monitor = LatencyMonitor(window_size=50, sla_ms=10)
router = ModelRouter(latency_monitor)

app = FastAPI()

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)

    model_choice = router.choose_model()
    model = large_model if model_choice == "large" else small_model

    start = time.time()
    prediction = model.predict(features)[0]
    latency_ms = (time.time() - start) * 1000

    latency_monitor.record(latency_ms)

    return {
        "prediction": int(prediction),
        "latency_ms": round(latency_ms, 2),
        "avg_latency_ms": round(latency_monitor.avg_latency(), 2),
        "p95_latency_ms": round(latency_monitor.p95_latency(), 2),
        "sla_violated": latency_monitor.sla_violated(),
        "model_used": model_choice
    }

@app.get("/metrics")
def metrics():
    return {
        "total_requests": latency_monitor.total_requests,
        "sla_violations": latency_monitor.sla_violations,
        "avg_latency_ms": round(latency_monitor.avg_latency(), 2),
        "p95_latency_ms": round(latency_monitor.p95_latency(), 2),
        "last_model_used": router.last_model
    }
