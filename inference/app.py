import time
import joblib
import numpy as np
import torch

from fastapi import FastAPI, Response
from pydantic import BaseModel

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from inference.latency import LatencyMonitor
from inference.router import ModelRouter
from models.torch_model import RiskNet


# ======================
# Load models (startup)
# ======================

# Sklearn models
large_sklearn = joblib.load("models/large_model.joblib")
small_model = joblib.load("models/small_model.joblib")

# Torch deep model
torch_model = RiskNet(input_dim=30)
torch_model.eval()


# ======================
# Monitoring + Routing
# ======================

latency_monitor = LatencyMonitor(
    window_size=50,
    sla_ms=10   # strict SLA to force routing under load
)

router = ModelRouter(latency_monitor)


# ======================
# Prometheus Metrics
# ======================

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests"
)

SLA_VIOLATIONS = Counter(
    "inference_sla_violations_total",
    "Total SLA violations"
)

INFERENCE_LATENCY = Histogram(
    "inference_latency_ms",
    "Inference latency in milliseconds",
    buckets=(5, 10, 20, 50, 100, 200)
)


# ======================
# FastAPI App
# ======================

app = FastAPI()


class InputData(BaseModel):
    features: list


# ======================
# Prediction Endpoint
# ======================

@app.post("/predict")
def predict(data: InputData):
    REQUEST_COUNT.inc()

    features = np.array(data.features).reshape(1, -1)

    model_choice = router.choose_model()

    start = time.time()

    if model_choice == "large":
        # 🔴 Artificial delay to simulate heavy deep model
        time.sleep(0.02)  # 20 ms

        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            prediction = int((torch_model(x) > 0.5).item())
    else:
        prediction = int(small_model.predict(features)[0])

    latency_ms = (time.time() - start) * 1000

    # Record metrics
    INFERENCE_LATENCY.observe(latency_ms)
    latency_monitor.record(latency_ms)

    if latency_monitor.sla_violated():
        SLA_VIOLATIONS.inc()

    return {
        "prediction": prediction,
        "latency_ms": round(latency_ms, 2),
        "avg_latency_ms": round(latency_monitor.avg_latency(), 2),
        "p95_latency_ms": round(latency_monitor.p95_latency(), 2),
        "sla_violated": latency_monitor.sla_violated(),
        "model_used": model_choice
    }


# ======================
# Prometheus Metrics Endpoint
# ======================

@app.get("/metrics_prom")
def prometheus_metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
