import time
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train large (slower) model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Measure inference latency
start = time.time()
time.sleep(0.05)
_ = model.predict(X_test[:1])
latency_ms = (time.time() - start) * 1000

print(f"Large Model Accuracy: {acc:.4f}")
print(f"Large Model Inference Latency: {latency_ms:.2f} ms")

joblib.dump(model, "models/large_model.joblib")
