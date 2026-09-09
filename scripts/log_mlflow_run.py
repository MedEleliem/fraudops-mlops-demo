import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from register_model import _parse_classification_report


ARTIFACTS = [
    "results/model_summary.txt",
    "results/classification_report_ns.txt",
    "results/classification_report_wd.txt",
    "results/before_resampling/Count_before_undersampling.png",
    "results/before_resampling/distrib_before_undersampling.png",
    "results/after_resampling/Distrib_after_undersampling.png",
    "results/after_resampling/Corr_matrices_comp.png",
    "results/confusionmatrix_ns.png",
    "results/confusionmatrix_wd.png",
    "results/logit.pkl",
    "results/finalvar.pkl",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Log the final DVC pipeline outputs to MLflow.")
    parser.add_argument("--experiment", default="Fraud_Credit_Card")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()

    try:
        import mlflow
    except ImportError as exc:
        raise SystemExit(
            "MLflow is not installed. Install it with: pip install -r requirements-mlflow.txt"
        ) from exc

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    params = _read_yaml(Path("params.yaml"))
    metrics = _collect_final_metrics()
    run_name = args.run_name or f"fraud-run-{_git_value(['rev-parse', '--short', 'HEAD'], 'local')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("git_commit", _git_value(["rev-parse", "HEAD"], "unknown"))
        mlflow.set_tag("source_branch", _git_value(["rev-parse", "--abbrev-ref", "HEAD"], "unknown"))
        mlflow.set_tag("pipeline", "dvc")
        mlflow.log_params(_flatten_params(params))
        mlflow.log_metrics(metrics)

        for artifact in ARTIFACTS:
            path = Path(artifact)
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path=_artifact_group(path))

        print(run.info.run_id)


def _collect_final_metrics() -> dict[str, float]:
    metrics: dict[str, float] = {}
    accuracy_ns = _read_json(Path("results/accuracy_ns.json"), default={})
    accuracy_wd = _read_json(Path("results/accuracy_wd.json"), default={})

    if "model_accuracy" in accuracy_ns:
        metrics["balanced_test_accuracy"] = float(accuracy_ns["model_accuracy"])
    if "model_accuracy" in accuracy_wd:
        metrics["full_dataset_accuracy"] = float(accuracy_wd["model_accuracy"])

    for prefix, report_path in {
        "balanced_test": Path("results/classification_report_ns.txt"),
        "full_dataset": Path("results/classification_report_wd.txt"),
    }.items():
        report_metrics = _parse_classification_report(report_path)
        for metric_name, value in report_metrics.items():
            metrics[f"{prefix}_{metric_name}"] = value
    return metrics


def _flatten_params(params: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(f"{prefix}.{child_key}" if prefix else str(child_key), child_value)
        else:
            flattened[prefix] = value

    walk("", params)
    return flattened


def _artifact_group(path: Path) -> str:
    if path.suffix == ".png":
        return "plots"
    if path.suffix == ".pkl":
        return "model"
    return "reports"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as yaml_file:
        return yaml.safe_load(yaml_file) or {}


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def _git_value(args: list[str], default: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return default


if __name__ == "__main__":
    main()
