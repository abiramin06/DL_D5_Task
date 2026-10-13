from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid_ticket():
    response = client.post(
        "/predict",
        json={"text": "I was charged twice"}
    )
    assert response.status_code == 200

    data = response.json()
    assert "category" in data
    assert "priority" in data
    assert "confidence" in data


def test_predict_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_missing_text():
    response = client.post("/predict", json={})
    assert response.status_code == 422