import json
import math
import os
import pickle
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class TransactionPayload(BaseModel):
    Time: float = 0
    Amount: float = Field(0, ge=0)
    V1: float = 0
    V2: float = 0
    V3: float = 0
    V4: float = 0
    V5: float = 0
    V6: float = 0
    V7: float = 0
    V8: float = 0
    V9: float = 0
    V10: float = 0
    V11: float = 0
    V12: float = 0
    V13: float = 0
    V14: float = 0
    V15: float = 0
    V16: float = 0
    V17: float = 0
    V18: float = 0
    V19: float = 0
    V20: float = 0
    V21: float = 0
    V22: float = 0
    V23: float = 0
    V24: float = 0
    V25: float = 0
    V26: float = 0
    V27: float = 0
    V28: float = 0


class ServingModel:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.model: Any | None = None
        self.features: list[str] = []
        self.manifest: dict[str, Any] = {}
        self.mode: str | None = None
        self.loaded_at: str | None = None
        self.last_reload_error: str | None = None
        self._signature: tuple[tuple[int, int], ...] | None = None
        self._lock = threading.RLock()
        self.load(force=True)

    @property
    def ready(self) -> bool:
        return self.model is not None and bool(self.features)

    def _artifact_signature(self) -> tuple[tuple[int, int], ...] | None:
        model_path = self.model_dir / "logit.pkl"
        features_path = self.model_dir / "finalvar.pkl"
        manifest_path = self.model_dir / "manifest.json"
        summary_path = self.model_dir / "model_summary.txt"
        if model_path.exists() and features_path.exists() and manifest_path.exists():
            paths = (model_path, features_path, manifest_path)
        elif summary_path.exists() and manifest_path.exists():
            paths = (summary_path, manifest_path)
        else:
            return None
        return tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in paths)

    def _load_pickle_artifacts(self) -> tuple[Any, list[str], dict[str, Any], str]:
        model_path = self.model_dir / "logit.pkl"
        features_path = self.model_dir / "finalvar.pkl"
        manifest_path = self.model_dir / "manifest.json"
        with model_path.open("rb") as model_file:
            loaded_model = pickle.load(model_file)
        with features_path.open("rb") as features_file:
            loaded_features = [str(feature) for feature in pickle.load(features_file)]
        loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return loaded_model, loaded_features, loaded_manifest, "pickle"

    def _load_summary_artifacts(self) -> tuple[dict[str, float], list[str], dict[str, Any], str]:
        summary_path = self.model_dir / "model_summary.txt"
        manifest_path = self.model_dir / "manifest.json"
        summary = summary_path.read_text(encoding="utf-8", errors="ignore")
        coefficients: dict[str, float] = {}
        for line in summary.splitlines():
            match = re.match(r"^\s*(const|V\d+|Time|Amount)\s+(-?\d+(?:\.\d+)?)\s+", line)
            if match:
                coefficients[match.group(1)] = float(match.group(2))
        loaded_features = [feature for feature in coefficients if feature != "const"]
        if "const" not in coefficients or not loaded_features:
            raise ValueError("No usable coefficients were found in model_summary.txt")
        loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        loaded_manifest.setdefault("threshold", 0.5)
        loaded_manifest["serving_mode"] = "summary_coefficients"
        return coefficients, loaded_features, loaded_manifest, "summary"

    def load(self, force: bool = False) -> bool:
        signature = self._artifact_signature()
        if signature is None or (not force and signature == self._signature):
            return False

        model_path = self.model_dir / "logit.pkl"
        features_path = self.model_dir / "finalvar.pkl"
        try:
            if model_path.exists() and features_path.exists():
                loaded_model, loaded_features, loaded_manifest, loaded_mode = self._load_pickle_artifacts()
            else:
                loaded_model, loaded_features, loaded_manifest, loaded_mode = self._load_summary_artifacts()
            if not loaded_features:
                raise ValueError("The approved model contains no serving features")
            if signature != self._artifact_signature():
                return False
        except Exception as exc:
            self.last_reload_error = str(exc)
            return False

        with self._lock:
            self.model = loaded_model
            self.features = loaded_features
            self.manifest = loaded_manifest
            self.mode = loaded_mode
            self._signature = signature
            self.loaded_at = datetime.now(timezone.utc).isoformat()
            self.last_reload_error = None
        return True

    def refresh_if_changed(self) -> bool:
        return self.load(force=False)

    def predict(self, payload: TransactionPayload) -> dict[str, Any]:
        self.refresh_if_changed()
        if not self.ready:
            raise RuntimeError("No approved serving model is deployed yet")

        row = payload.model_dump()
        with self._lock:
            model = self.model
            features = list(self.features)
            manifest = dict(self.manifest)
            mode = self.mode
        if mode == "summary":
            score = float(model.get("const", 0.0))
            for feature in features:
                score += float(model.get(feature, 0.0)) * float(row.get(feature, 0.0))
            probability = 1 / (1 + math.exp(-max(min(score, 700), -700)))
        else:
            import pandas as pd
            import statsmodels.api as sm

            frame = pd.DataFrame([{feature: row.get(feature, 0.0) for feature in FEATURES}])
            selected = sm.add_constant(frame[features], has_constant="add")
            probability = float(model.predict(selected).iloc[0])
        threshold = float(manifest.get("threshold", 0.5))
        prediction = int(probability >= threshold)
        return {
            "prediction": prediction,
            "label": "Fraud" if prediction else "Not Fraud",
            "fraud_probability": probability,
            "threshold": threshold,
            "model_version": manifest.get("version", self.model_dir.name),
            "serving_mode": mode,
            "features_used": features,
        }


def create_app() -> FastAPI:
    app = FastAPI(title="FraudOps Serving API")
    model_dir = Path(os.getenv("FRAUD_SERVING_MODEL_DIR", "models/production"))
    serving_model = ServingModel(model_dir)
    templates = Jinja2Templates(directory="serving_app/templates")
    app.mount("/static", StaticFiles(directory="serving_app/static"), name="serving_static")

    @app.get("/health")
    def health() -> dict[str, Any]:
        reloaded = serving_model.refresh_if_changed()
        return {
            "status": "ok" if serving_model.ready else "waiting_for_model",
            "model_ready": serving_model.ready,
            "model_dir": str(model_dir),
            "model_version": serving_model.manifest.get("version") if serving_model.ready else None,
            "loaded_at": serving_model.loaded_at,
            "reloaded": reloaded,
            "reload_error": serving_model.last_reload_error,
        }

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        serving_model.refresh_if_changed()
        if not serving_model.ready:
            raise HTTPException(status_code=503, detail="No approved serving model is deployed yet")
        return {
            "model_version": serving_model.manifest.get("version", model_dir.name),
            "features": serving_model.features,
            "manifest": serving_model.manifest,
            "loaded_at": serving_model.loaded_at,
        }

    @app.post("/reload")
    def reload_model() -> dict[str, Any]:
        reloaded = serving_model.load(force=True)
        return {
            "reloaded": reloaded,
            "model_ready": serving_model.ready,
            "model_version": serving_model.manifest.get("version") if serving_model.ready else None,
            "loaded_at": serving_model.loaded_at,
            "error": serving_model.last_reload_error,
        }

    @app.post("/predict")
    def predict(payload: TransactionPayload) -> dict[str, Any]:
        try:
            return serving_model.predict(payload)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        serving_model.refresh_if_changed()
        return templates.TemplateResponse(
            request,
            "predict.html",
            {
                "features": FEATURES,
                "model_ready": serving_model.ready,
                "model_version": serving_model.manifest.get("version", "not deployed"),
            },
        )

    return app


app = create_app()
