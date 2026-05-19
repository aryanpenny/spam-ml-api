from fastapi.testclient import TestClient
from app.main import app

def test_health_returns_ok():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_predict_returns_label_and_score():
    client = TestClient(app)
    payload = {"message": "Hello, this is a normal message"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "label" in body
    assert "score" in body