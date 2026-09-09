import os
from pathlib import Path
from typing import Any

from app.ml.diff import metric_key
from app.ml.schemas import ModelDiff, ModelMetric, ModelRegistry, ModelVersion


PRIMARY_METRICS = [
    ("balanced_test:fraud_recall:1", "Balanced recall"),
    ("balanced_test:fraud_precision:1", "Balanced precision"),
    ("balanced_test:fraud_f1:1", "Balanced F1"),
    ("full_dataset:fraud_recall:1", "Full recall"),
    ("full_dataset:fraud_precision:1", "Full precision"),
    ("full_dataset:fraud_f1:1", "Full F1"),
    ("full_dataset:accuracy:all", "Full accuracy"),
]


def dashboard_context(
    registry: ModelRegistry,
    selected: ModelVersion,
    baseline: ModelVersion,
    diff: ModelDiff,
) -> dict[str, Any]:
    comparison_metrics = _comparison_metrics(diff)
    return {
        "model_rows": [_model_row(version) for version in registry.versions],
        "selected_card": _model_card(selected),
        "baseline_card": _model_card(baseline),
        "comparison_metrics": comparison_metrics,
        "comparison_stats": _comparison_stats(comparison_metrics, diff),
        "score_radar": _score_radar(selected, baseline),
        "metric_history": _metric_history(registry),
        "mlflow": mlflow_run_summary(selected.version),
    }


def _comparison_stats(metrics: list[dict[str, Any]], diff: ModelDiff) -> dict[str, Any]:
    comparable = [metric for metric in metrics if metric["delta"] is not None]
    improved = [metric for metric in comparable if metric["delta"] > 0.000001]
    regressed = [metric for metric in comparable if metric["delta"] < -0.000001]
    stable = len(comparable) - len(improved) - len(regressed)
    average_delta = (
        sum(metric["delta"] for metric in comparable) / len(comparable)
        if comparable
        else 0.0
    )
    largest_gain = max(comparable, key=lambda metric: metric["delta"], default=None)
    largest_loss = min(comparable, key=lambda metric: metric["delta"], default=None)
    artifact_ready = sum(
        1 for artifact in diff.artifact_diffs if artifact.after_exists is True
    )
    return {
        "improved": len(improved),
        "regressed": len(regressed),
        "stable": stable,
        "average_delta": average_delta,
        "largest_gain": largest_gain,
        "largest_loss": largest_loss,
        "param_changes": len(diff.param_diffs),
        "feature_changes": len(diff.feature_diff.added) + len(diff.feature_diff.removed),
        "artifact_ready": artifact_ready,
        "artifact_total": len(diff.artifact_diffs),
        "artifact_pct": round(
            artifact_ready / len(diff.artifact_diffs) * 100, 1
        ) if diff.artifact_diffs else 100.0,
    }


def mlflow_run_summary(version: str) -> dict[str, Any]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    if tracking_uri.startswith("sqlite:///"):
        db_path = Path(tracking_uri.replace("sqlite:///", "", 1))
        if not db_path.exists():
            return _empty_mlflow(version, tracking_uri, "tracking database not found")

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        return _empty_mlflow(version, tracking_uri, "mlflow package not installed")

    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment_ids = [experiment.experiment_id for experiment in client.search_experiments()]
        runs = client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=f"attributes.run_name = '{version}'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
    except Exception as exc:
        return _empty_mlflow(version, tracking_uri, str(exc))

    if not runs:
        return _empty_mlflow(
            version,
            tracking_uri,
            "No MLflow run with this snapshot version as run_name was found.",
        )

    run = runs[0]
    return {
        "available": True,
        "tracking_uri": tracking_uri,
        "run_name": run.info.run_name,
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "artifact_uri": run.info.artifact_uri,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": dict(run.data.tags),
    }


def _empty_mlflow(version: str, tracking_uri: str, reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "tracking_uri": tracking_uri,
        "run_name": version,
        "reason": reason,
        "params": {},
        "metrics": {},
        "tags": {},
    }


def _model_row(version: ModelVersion) -> dict[str, Any]:
    metrics = _metric_map(version.metrics)
    artifacts_ready = sum(1 for artifact in version.artifacts if artifact.exists)
    serving_ready = all(
        artifact.exists for artifact in version.artifacts if artifact.required_for_serving
    )
    return {
        "version": version.version,
        "name": version.name,
        "stage": version.stage,
        "status": version.status,
        "trained_at": version.trained_at,
        "threshold": version.threshold,
        "p_value": version.params.get("train_model", {}).get("p-value"),
        "features_count": version.features_count,
        "artifacts_ready": artifacts_ready,
        "artifacts_total": len(version.artifacts),
        "serving_ready": serving_ready,
        "recall": metrics.get("full_dataset:fraud_recall:1"),
        "precision": metrics.get("full_dataset:fraud_precision:1"),
        "f1": metrics.get("full_dataset:fraud_f1:1"),
        "accuracy": metrics.get("full_dataset:accuracy:all"),
    }


def _model_card(version: ModelVersion) -> dict[str, Any]:
    metrics = _metric_map(version.metrics)
    return {
        "version": version.version,
        "name": version.name,
        "stage": version.stage,
        "status": version.status,
        "algorithm": version.algorithm,
        "trained_at": version.trained_at,
        "training_data": version.training_data,
        "git_commit": version.git_commit,
        "source_branch": version.source_branch,
        "threshold": version.threshold,
        "params": _flatten_params(version.params),
        "metrics": [
            {"key": key, "label": label, "value": metrics.get(key)}
            for key, label in PRIMARY_METRICS
        ],
        "features": version.features,
        "artifacts": version.artifacts,
        "risks": version.risks,
        "notes": version.notes,
    }


def _comparison_metrics(diff: ModelDiff) -> list[dict[str, Any]]:
    rows = []
    for key, label in PRIMARY_METRICS:
        item = diff.metric_diffs.get(key)
        if not item:
            continue
        rows.append(
            {
                "key": key,
                "label": label,
                "before": item.before,
                "after": item.after,
                "delta": item.delta,
                "changed": item.changed,
                "before_pct": _percent(item.before),
                "after_pct": _percent(item.after),
            }
        )
    return rows


def _score_radar(selected: ModelVersion, baseline: ModelVersion) -> list[dict[str, Any]]:
    selected_metrics = _metric_map(selected.metrics)
    baseline_metrics = _metric_map(baseline.metrics)
    keys = [
        ("full_dataset:fraud_recall:1", "Recall"),
        ("full_dataset:fraud_precision:1", "Precision"),
        ("full_dataset:fraud_f1:1", "F1"),
        ("full_dataset:accuracy:all", "Accuracy"),
    ]
    return [
        {
            "label": label,
            "selected": selected_metrics.get(key),
            "baseline": baseline_metrics.get(key),
            "selected_pct": _percent(selected_metrics.get(key)),
            "baseline_pct": _percent(baseline_metrics.get(key)),
        }
        for key, label in keys
    ]


def _metric_history(registry: ModelRegistry) -> list[dict[str, Any]]:
    rows = []
    for version in reversed(registry.versions):
        metrics = _metric_map(version.metrics)
        rows.append(
            {
                "version": version.version,
                "trained_at": version.trained_at,
                "recall": metrics.get("full_dataset:fraud_recall:1"),
                "precision": metrics.get("full_dataset:fraud_precision:1"),
                "f1": metrics.get("full_dataset:fraud_f1:1"),
                "accuracy": metrics.get("full_dataset:accuracy:all"),
            }
        )
    return rows


def _metric_map(metrics: list[ModelMetric]) -> dict[str, float]:
    return {metric_key(metric): metric.value for metric in metrics}


def _flatten_params(params: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(f"{prefix}.{child_key}" if prefix else str(child_key), child_value)
        else:
            rows.append({"key": prefix, "value": value})

    walk("", params)
    return rows


def _percent(value: Any) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(100.0, float(value) * 100.0))
