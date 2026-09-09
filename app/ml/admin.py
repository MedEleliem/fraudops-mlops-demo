import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.ml.registry import ModelRegistryStore
from app.ml.schemas import AdminState, ModelRegistry, ModelVersion, ParameterDraft, TrainingJob


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AdminStateStore:
    def __init__(self, state_path: Path, registry_store: ModelRegistryStore) -> None:
        self.state_path = state_path
        self.registry_store = registry_store

    def load(self) -> AdminState:
        if not self.state_path.exists():
            return AdminState()
        with self.state_path.open("r", encoding="utf-8") as state_file:
            return AdminState.model_validate(json.load(state_file))

    def save(self, state: AdminState) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("w", encoding="utf-8") as state_file:
            json.dump(state.model_dump(), state_file, indent=2)
            state_file.write("\n")

    def create_draft(self, payload: dict[str, Any]) -> ParameterDraft:
        registry = self.registry_store.load()
        active = (
            self.registry_store.get_version(registry.active_version, registry=registry)
            if registry.active_version
            else None
        )
        baseline_version = payload.get("base_version") or self.latest_submitted_version(
            registry,
            fallback=active.version if active else None,
        )
        created_at = utc_now()
        draft = ParameterDraft(
            id=f"draft-{created_at.replace(':', '').replace('-', '')}",
            name=payload.get("name") or "Parameter change",
            status="draft",
            created_at=created_at,
            base_version=baseline_version,
            params=payload["params"],
            threshold=payload.get("threshold", active.threshold if active else 0.5),
            note=payload.get("note", ""),
        )
        state = self.load()
        state.drafts.insert(0, draft)
        self.save(state)
        return draft

    def submit_retraining(
        self,
        draft_id: str,
        run_pipeline: bool = True,
    ) -> tuple[ParameterDraft, TrainingJob, ModelVersion]:
        state = self.load()
        draft = self._get_draft(state, draft_id)
        registry = self.registry_store.load()
        baseline = (
            self.registry_store.get_version(draft.base_version, registry=registry)
            if draft.base_version
            else None
        )
        created_at = utc_now()
        candidate_version = f"fraud-logit-{created_at[:10]}-{draft.id[-6:]}"
        candidate = self._candidate_from_draft(baseline, draft, candidate_version, created_at)

        draft.status = "submitted"
        draft.candidate_version = candidate.version

        job = TrainingJob(
            id=f"job-{created_at.replace(':', '').replace('-', '')}",
            draft_id=draft.id,
            status="training",
            created_at=created_at,
            message="Retraining started. DVC is reproducing the pipeline, then the registry and MLflow run will be updated.",
            candidate_version=candidate.version,
        )
        state.jobs.insert(0, job)
        self.save(state)

        if run_pipeline:
            try:
                self._write_params_yaml(draft.params)
                self._run_pipeline(candidate.version, draft.threshold)
                registry = self.registry_store.load()
                candidate = self.registry_store.get_version(candidate.version, registry=registry)
                draft.status = "completed"
                job.status = "completed"
                job.message = "Retraining completed. Final metrics, artifacts, registry entry, and MLflow run are available."
                self.save(state)
            except Exception as exc:
                draft.status = "blocked"
                job.status = "blocked"
                job.message = f"Retraining failed: {exc}"
                self.save(state)
                self._append_candidate(registry, candidate)
        else:
            self._append_candidate(registry, candidate)
        return draft, job, candidate

    def latest_submitted_version(
        self,
        registry: ModelRegistry | None = None,
        fallback: str | None = None,
    ) -> str | None:
        registry = registry or self.registry_store.load()
        registered_versions = {version.version for version in registry.versions}
        for draft in self.load().drafts:
            if draft.candidate_version and draft.candidate_version in registered_versions:
                return draft.candidate_version
        return fallback or registry.active_version

    def approve_model(self, version_id: str, production_dir: Path = Path("models/production")) -> ModelVersion:
        registry = self.registry_store.load()
        selected = self.registry_store.get_version(version_id, registry=registry)
        version_dir = Path("models/versions") / version_id
        required = [version_dir / "logit.pkl", version_dir / "finalvar.pkl", version_dir / "manifest.json"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise ValueError(f"Cannot approve model; missing serving artifacts: {', '.join(missing)}")

        if production_dir.exists():
            archive_dir = production_dir.with_name(f"{production_dir.name}_previous")
            if archive_dir.exists():
                shutil.rmtree(archive_dir)
            shutil.move(str(production_dir), str(archive_dir))
        production_dir.mkdir(parents=True, exist_ok=True)

        for artifact in selected.artifacts:
            source = Path(artifact.path)
            if source.exists():
                shutil.copy2(source, production_dir / source.name)

        manifest_path = production_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        manifest["version"] = version_id
        manifest["serving_image"] = f"fraud-serving:{version_id}"
        manifest["threshold"] = selected.threshold
        manifest["approved_at"] = utc_now()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        registry.active_version = version_id
        for version in registry.versions:
            if version.version == version_id:
                version.stage = "production"
                version.status = "approved"
                version.notes = f"{version.notes}\nApproved for client serving image fraud-serving:{version_id}.".strip()
            elif version.stage == "production":
                version.stage = "archived"
                version.status = "retired"
        with self.registry_store.registry_path.open("w", encoding="utf-8") as registry_file:
            json.dump(registry.model_dump(), registry_file, indent=2)
            registry_file.write("\n")
        return self.registry_store.get_version(version_id)

    def _get_draft(self, state: AdminState, draft_id: str) -> ParameterDraft:
        for draft in state.drafts:
            if draft.id == draft_id:
                return draft
        raise ValueError(f"Draft {draft_id!r} not found")

    def _candidate_from_draft(
        self,
        baseline: ModelVersion | None,
        draft: ParameterDraft,
        version: str,
        created_at: str,
    ) -> ModelVersion:
        candidate_payload = deepcopy(baseline.model_dump()) if baseline else _empty_candidate_payload()
        candidate_payload.update(
            {
                "version": version,
                "name": draft.name,
                "stage": "development",
                "status": "candidate",
                "trained_at": created_at,
                "source_branch": "platform-admin",
                "git_commit": "pending-ci",
                "params": draft.params,
                "threshold": draft.threshold,
                "artifacts": [
                    {
                        "name": "manifest",
                        "path": f"models/versions/{version}/manifest.json",
                        "required_for_serving": False,
                    },
                    {
                        "name": "params",
                        "path": f"models/versions/{version}/params.yaml",
                        "required_for_serving": False,
                    },
                    {
                        "name": "metrics",
                        "path": f"models/versions/{version}/metrics.json",
                        "required_for_serving": False,
                    },
                    {
                        "name": "model_pickle",
                        "path": f"models/versions/{version}/logit.pkl",
                        "required_for_serving": True,
                    },
                    {
                        "name": "selected_features_pickle",
                        "path": f"models/versions/{version}/finalvar.pkl",
                        "required_for_serving": True,
                    },
                ],
                "risks": [
                    {
                        "level": "medium",
                        "message": "Candidate created from platform parameters. Awaiting CI/DVC generated artifacts.",
                    }
                ],
                "notes": draft.note or "Parameter snapshot submitted from the admin platform.",
            }
        )
        candidate_payload["metrics"] = _simulated_candidate_metrics(
            baseline.metrics if baseline else [],
            draft.params,
            draft.threshold,
        )
        return ModelVersion.model_validate(candidate_payload)

    def _append_candidate(self, registry: ModelRegistry, candidate: ModelVersion) -> None:
        registry.versions = [
            version for version in registry.versions if version.version != candidate.version
        ]
        registry.versions.insert(0, candidate)
        registry_path = self.registry_store.registry_path
        with registry_path.open("w", encoding="utf-8") as registry_file:
            json.dump(registry.model_dump(), registry_file, indent=2)
            registry_file.write("\n")

    def _write_params_yaml(self, params: dict[str, Any]) -> None:
        import yaml

        with Path("params.yaml").open("w", encoding="utf-8") as params_file:
            yaml.safe_dump(params, params_file, sort_keys=False)

    def _run_pipeline(self, version: str, threshold: float) -> None:
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
        env.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        env.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
        env.setdefault("DVC_SITE_CACHE_DIR", str(Path(".cache/dvc/site").resolve()))
        env.setdefault("DVC_GLOBAL_CONFIG_DIR", str(Path(".config/dvc").resolve()))
        env.setdefault("DVC_SYSTEM_CONFIG_DIR", str(Path(".config/dvc-system").resolve()))
        env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        Path(env["DVC_SITE_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(env["DVC_GLOBAL_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(env["DVC_SYSTEM_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "dvc",
                "repro",
            ],
            env=env,
        )
        subprocess.check_call(
            [
                sys.executable,
                "scripts/register_model.py",
                "--version",
                version,
                "--copy-artifacts",
                "--stage",
                "development",
                "--status",
                "candidate",
                "--threshold",
                str(threshold),
            ],
            env=env,
        )
        subprocess.check_call(
            [
                sys.executable,
                "scripts/log_mlflow_run.py",
                "--run-name",
                version,
            ],
            env=env,
        )


def _simulated_candidate_metrics(
    baseline_metrics: list[Any],
    params: dict[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    metrics = [metric.model_dump() for metric in baseline_metrics]
    p_value = float(params.get("train_model", {}).get("p-value", 0.05))
    threshold_delta = 0.5 - threshold
    f1_lift = max(0.0, (0.05 - p_value) * 0.8) + threshold_delta * 0.08
    recall_lift = threshold_delta * 0.12
    precision_lift = max(0.0, (0.05 - p_value) * 0.25)

    for metric in metrics:
        if metric["name"] == "fraud_f1" and metric["split"] == "full_dataset":
            metric["value"] = round(min(1.0, metric["value"] + f1_lift), 4)
        if metric["name"] == "fraud_recall" and metric["split"] == "full_dataset":
            metric["value"] = round(min(1.0, metric["value"] + recall_lift), 4)
        if metric["name"] == "fraud_precision" and metric["split"] == "full_dataset":
            metric["value"] = round(min(1.0, metric["value"] + precision_lift), 4)
        if metric["name"] == "accuracy" and metric["split"] == "full_dataset":
            metric["value"] = round(max(0.0, metric["value"] - abs(threshold_delta) * 0.02), 4)
    return metrics


def _empty_candidate_payload() -> dict[str, Any]:
    return {
        "version": "pending",
        "name": "Initial candidate",
        "stage": "development",
        "status": "candidate",
        "algorithm": "statsmodels.Logit",
        "trained_at": utc_now(),
        "training_data": "creditcard.csv with DVC pipeline",
        "git_commit": "pending-ci",
        "source_branch": "platform-admin",
        "params": {},
        "threshold": 0.5,
        "features_count": 0,
        "features": [],
        "metrics": [],
        "artifacts": [],
        "risks": [],
        "notes": "",
    }
