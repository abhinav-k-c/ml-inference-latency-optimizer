import time
import joblib
import numpy as np
import torch

from fastapi import FastAPI, Response
from pydantic import BaseModel

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from inference.latency import LatencyMonitor
from inference.router import ModelRouter
from models.torch_model import RiskNet
from utils.config_loader import load_config


# ======================
# Load typed config
# ======================

config = load_config()

SLA_MS = config.routing.sla_ms
WINDOW_SIZE = config.routing.window_size
DEBUG_MODE = config.routing.debug_mode

ARTIFICIAL_DELAY_MS = config.models["large"].artificial_delay_ms


# ======================
# Load models
# ======================

small_model = joblib.load(config.models["small"].path)

torch_model = RiskNet(input_dim=30)
torch_model.eval()


# ======================
# Monitoring + Routing
# ======================

latency_monitor = LatencyMonitor(
    window_size=WINDOW_SIZE,
    sla_ms=SLA_MS,
)

router = ModelRouter(latency_monitor)


# ======================
# Prometheus Metrics
# ======================

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests",
)

SLA_VIOLATIONS = Counter(
    "inference_sla_violations_total",
    "Total SLA violations",
)

INFERENCE_LATENCY = Histogram(
    "inference_latency_ms",
    "Inference latency in milliseconds",
    buckets=(5, 10, 20, 50, 100, 200),
)


# ======================
# FastAPI App
# ======================

app = FastAPI(title="SLA-Aware ML Inference Service")


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
        if ARTIFICIAL_DELAY_MS > 0:
            time.sleep(ARTIFICIAL_DELAY_MS / 1000)

        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            prediction = int((torch_model(x) > 0.5).item())
    else:
        prediction = int(small_model.predict(features)[0])

    latency_ms = (time.time() - start) * 1000

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
        "model_used": model_choice,
    }


# ======================
# Metrics Endpoint
# ======================

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
