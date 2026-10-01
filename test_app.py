import csv
import os

os.environ["GEMINI_API_KEY"] = ""  # keep tests offline; load_dotenv won't override a set var

from fastapi.testclient import TestClient  # noqa: E402

from app import app  # noqa: E402

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_missing_features_are_filled_with_medians():
    r = client.post("/predict", json={"TransactionAmt": 150.0})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["risk_score"] <= 100
    assert body["ai_explanation"]


def test_non_numeric_feature_is_rejected():
    r = client.post("/predict", json={"TransactionAmt": "abc"})
    assert r.status_code == 422


def test_known_fraud_rows_are_flagged():
    with open("demo_fraud_data.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    for row in rows:
        assert client.post("/predict", json=row).json()["is_fraud"]
