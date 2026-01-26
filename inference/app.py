import time
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from inference.latency import LatencyMonitor


# -----------------------------
# Load models once at startup
# -----------------------------
large_model = joblib.load("models/large_model.joblib")
small_model = joblib.load("models/small_model.joblib")


# -----------------------------
# Latency monitor with SLA
# -----------------------------
latency_monitor = LatencyMonitor(
    window_size=50,   # rolling window
    sla_ms=50         # SLA threshold in ms
)


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="ML Inference Latency Optimizer")


class InputData(BaseModel):
    features: list


@app.post("/predict")
def predict(data: InputData):
    # Prepare input
    features = np.array(data.features).reshape(1, -1)

    # -----------------------------
    # Dynamic model selection
    # -----------------------------
    if latency_monitor.sla_violated():
        model = small_model
        model_name = "small"
    else:
        model = large_model
        model_name = "large"

    # -----------------------------
    # Inference + latency tracking
    # -----------------------------
    start = time.time()
    prediction = model.predict(features)[0]
    latency_ms = (time.time() - start) * 1000

    # Record latency AFTER inference
    latency_monitor.record(latency_ms)

    # -----------------------------
    # Response
    # -----------------------------
    return {
        "prediction": int(prediction),
        "latency_ms": round(latency_ms, 2),
        "avg_latency_ms": round(latency_monitor.avg_latency(), 2),
        "p95_latency_ms": round(latency_monitor.p95_latency(), 2),
        "sla_violated": latency_monitor.sla_violated(),
        "model_used": model_name
    }
