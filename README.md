🚀 ML Inference Latency Optimizer
SLA-Aware Model Routing for Real-Time ML Systems

A production-style machine learning inference service that dynamically routes traffic between models based on real-time latency and SLA compliance.

This project simulates how real ML platforms (ads ranking, fraud detection, recommendation systems) balance accuracy vs latency under strict performance constraints.

🎯 Project Goal

In real-world ML systems:

Larger models → Higher accuracy → Higher latency

Smaller models → Lower accuracy → Faster response

This project builds a dynamic inference service that:

✔ Monitors real-time latency
✔ Computes rolling p95 latency
✔ Detects SLA violations
✔ Automatically switches models when SLA is breached

🧠 Architecture Overview
Client Request
      ↓
FastAPI Inference Service
      ↓
Latency Monitor (Rolling Window)
      ↓
Model Router (SLA-Based Decision)
      ↓
Large Model (Torch)  OR  Small Model (Sklearn)
      ↓
Prediction + Metrics

⚙️ Tech Stack

Python 3.14

FastAPI

PyTorch

Scikit-learn

Pydantic (Typed Config Schema)

Prometheus Metrics

YAML Config-Driven Architecture

📂 Project Structure
ml-inference-latency-optimizer/
│
├── inference/
│   ├── app.py
│   ├── latency.py
│   └── router.py
│
├── models/
│   ├── torch_model.py
│   ├── train_large_model.py
│   └── train_small_model.py
│
├── config/
│   ├── config.yaml
│   └── schema.py
│
├── utils/
│   ├── config_loader.py
│   └── timing.py
│
├── load_test.py
├── requirements.txt
└── README.md

🔄 SLA-Based Routing Logic

Routing Strategy:

If p95 latency > SLA threshold
→ Switch to small model

Otherwise
→ Use large model

Configured via config.yaml:

routing:
  strategy: sla_based
  sla_ms: 10
  window_size: 50
  debug_mode: true

📊 Metrics Exposed

Prometheus metrics endpoint:

GET /metrics


Tracked metrics:

Total requests

SLA violations

Inference latency histogram

Rolling average latency

Rolling p95 latency

🚀 Run Locally

Activate virtual environment:

source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Start server:

python -m uvicorn inference.app:app --reload


Visit:

http://127.0.0.1:8000/docs

🧪 Example Request
{
  "features": [0.1, 0.2, 0.3, ..., 0.5]
}


Example Response:

{
  "prediction": 1,
  "latency_ms": 37.08,
  "avg_latency_ms": 10.14,
  "p95_latency_ms": 27.22,
  "sla_violated": true,
  "model_used": "small"
}

📈 What This Project Demonstrates

This project showcases:

• ML systems thinking
• Latency-aware architecture
• SLA-driven dynamic routing
• Config-driven design
• Typed configuration with Pydantic
• Observability integration (Prometheus)
• Production-style FastAPI service



👨‍💻 Author

Abhinav Reddy Kurri
MS Computer Science
Machine Learning & ML Systems
