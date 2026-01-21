import time
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train small (fast) model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Measure inference latency
start = time.time()
_ = model.predict(X_test[:1])
latency_ms = (time.time() - start) * 1000

print(f"Small Model Accuracy: {acc:.4f}")
print(f"Small Model Inference Latency: {latency_ms:.2f} ms")

joblib.dump(model, "models/small_model.joblib")
