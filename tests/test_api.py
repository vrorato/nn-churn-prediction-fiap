import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from nn_churn_prediction_fiap.api.dependencies import get_predictor
from nn_churn_prediction_fiap.api.main import app

VALID_PAYLOAD = {
    "tenure": 12,
    "MonthlyCharges": 69.0,
    "TotalCharges": 830.0,
    "gender": "Male",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "PaperlessBilling": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
}


@pytest.fixture()
def client(trained_model: Pipeline) -> TestClient:
    app.dependency_overrides[get_predictor] = lambda: trained_model
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_valid(client: TestClient) -> None:
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["churn_flag"], bool)
    assert "threshold" in data


def test_predict_invalid_payload(client: TestClient) -> None:
    resp = client.post("/predict", json={"tenure": "not_a_number"})
    assert resp.status_code == 422
