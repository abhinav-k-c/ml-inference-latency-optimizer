import requests
import random
import concurrent.futures

URL = "http://127.0.0.1:8000/predict"

payload = {
    "features": [
        17.99, 10.38, 122.8, 1001.0, 0.1184,
        0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
        1.095, 0.9053, 8.589, 153.4, 0.006399,
        0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
        25.38, 17.33, 184.6, 2019.0, 0.1622,
        0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]
}

def send_request():
    r = requests.post(URL, json=payload)
    return r.json()["model_used"]

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(lambda _: send_request(), range(100)))

    print("Large model used:", results.count("large"))
    print("Small model used:", results.count("small"))
