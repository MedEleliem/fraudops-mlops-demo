import json

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    registry_path = tmp_path / "registry.json"
    admin_path = tmp_path / "admin_state.json"
    registry_path.write_text('{"active_version": null, "versions": []}\n')
    admin_path.write_text('{"drafts": [], "jobs": []}\n')
    monkeypatch.setenv("FRAUD_API_REGISTRY_PATH", str(registry_path))
    monkeypatch.setenv("FRAUD_API_ADMIN_STATE_PATH", str(admin_path))
    get_settings.cache_clear()
    yield {"registry": registry_path, "admin": admin_path}
    get_settings.cache_clear()


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_active_model_metadata():
    response = client.get("/api/models/active")
    payload = response.json()

    assert response.status_code == 404
    assert payload["detail"] == "No production model has been approved yet"


def test_model_version_metadata():
    response = client.get("/api/models/fraud-logit-2021-05-18-candidate")
    payload = response.json()

    assert response.status_code == 404
    assert "is not listed" in payload["detail"]


def test_model_diff_endpoint():
    response = client.get(
        "/api/models/fraud-logit-2021-05-18-candidate/diff/fraud-logit-2021-05-18"
    )
    payload = response.json()

    assert response.status_code == 404
    assert payload["detail"] == "No model versions are registered yet"


def test_registry_page_renders():
    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "Aucun modèle enregistré" in response.text
    assert "Premier run" in response.text
    assert "Cycle de promotion" in response.text


def test_standalone_project_presentation_renders():
    response = client.get("/presentation")

    assert response.status_code == 200
    assert "Transformer un modèle ML en service gouverné" in response.text
    assert "Le laboratoire ne touche jamais le service client" in response.text


def test_first_candidate_renders_without_production(isolated_registry):
    candidate = {
        "version": "fraud-logit-first-run",
        "name": "First real run",
        "stage": "development",
        "status": "candidate",
        "algorithm": "statsmodels.Logit",
        "trained_at": "2026-08-06T14:14:31Z",
        "training_data": "creditcard.csv",
        "params": {"train_model": {"p-value": 0.05}},
        "threshold": 0.5,
        "features_count": 2,
        "features": ["V1", "Amount"],
        "metrics": [
            {"name": "fraud_recall", "value": 0.89, "split": "full_dataset", "target_class": "1"},
            {"name": "fraud_precision", "value": 0.16, "split": "full_dataset", "target_class": "1"},
            {"name": "fraud_f1", "value": 0.27, "split": "full_dataset", "target_class": "1"},
            {"name": "accuracy", "value": 0.99, "split": "full_dataset"},
        ],
        "artifacts": [],
    }
    isolated_registry["registry"].write_text(
        json.dumps({"active_version": None, "versions": [candidate]})
    )

    dashboard = client.get("/")
    control = client.get("/dashboard")
    runs = client.get("/runs")

    assert dashboard.status_code == 200
    assert "Transformer un modèle ML en service gouverné" in dashboard.text
    assert control.status_code == 200
    assert "Modèles de détection de fraude" in control.text
    assert runs.status_code == 200
    assert "Runs & différences" in runs.text
