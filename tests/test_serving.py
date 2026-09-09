import json
import os

import pytest
from fastapi.testclient import TestClient

from serving_app.main import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    empty_model_dir = tmp_path / "production"
    empty_model_dir.mkdir()
    monkeypatch.setenv("FRAUD_SERVING_MODEL_DIR", str(empty_model_dir))
    return TestClient(create_app())


def test_serving_health_waits_for_approved_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "waiting_for_model"
    assert response.json()["model_ready"] is False


def test_prediction_ui_renders_without_model(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Nouvelle transaction" in response.text
    assert "Modèle en attente" in response.text


def test_predict_requires_deployed_model(client):
    response = client.post("/predict", json={"Time": 0, "Amount": 0})

    assert response.status_code == 503
    assert "No approved serving model" in response.json()["detail"]


def test_serving_uses_summary_coefficients_for_portfolio_demo(tmp_path, monkeypatch):
    model_dir = tmp_path / "production"
    model_dir.mkdir()
    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"version": "model-v1", "threshold": 0.5}))
    (model_dir / "model_summary.txt").write_text(
        """
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -4.0000      0.049    -134.412      0.000      -6.728      -6.535
Amount         0.5000      0.000      41.133      0.000       0.015       0.017
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("FRAUD_SERVING_MODEL_DIR", str(model_dir))
    deployed_client = TestClient(create_app())

    assert deployed_client.get("/health").json()["model_version"] == "model-v1"

    manifest_path.write_text(json.dumps({"version": "model-v2", "threshold": 0.3}))
    stat = manifest_path.stat()
    os.utime(manifest_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    health = deployed_client.get("/health").json()
    prediction = deployed_client.post("/predict", json={"Amount": 10}).json()

    assert health["reloaded"] is True
    assert health["model_version"] == "model-v2"
    assert prediction["model_version"] == "model-v2"
    assert prediction["threshold"] == 0.3
    assert prediction["serving_mode"] == "summary"
    assert prediction["label"] == "Fraud"
