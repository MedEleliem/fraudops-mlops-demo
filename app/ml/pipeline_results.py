import json
from pathlib import Path
from typing import Any


PLOTS = [
    {
        "id": "count_before",
        "title": "Class count before resampling",
        "stage": "understand data",
        "path": "results/before_resampling/Count_before_undersampling.png",
    },
    {
        "id": "distribution_before",
        "title": "Feature distribution before resampling",
        "stage": "understand data",
        "path": "results/before_resampling/distrib_before_undersampling.png",
    },
    {
        "id": "distribution_after",
        "title": "Distribution after resampling",
        "stage": "resample data",
        "path": "results/after_resampling/Distrib_after_undersampling.png",
    },
    {
        "id": "correlation_after",
        "title": "Correlation matrices after resampling",
        "stage": "resample data",
        "path": "results/after_resampling/Corr_matrices_comp.png",
    },
    {
        "id": "confusion_balanced",
        "title": "Confusion matrix on balanced test sample",
        "stage": "test model",
        "path": "results/confusionmatrix_ns.png",
    },
    {
        "id": "confusion_full",
        "title": "Confusion matrix on full dataset",
        "stage": "test model",
        "path": "results/confusionmatrix_wd.png",
    },
]

REPORTS = [
    {
        "title": "Classification report, balanced sample",
        "path": "results/classification_report_ns.txt",
    },
    {
        "title": "Classification report, full dataset",
        "path": "results/classification_report_wd.txt",
    },
    {
        "title": "Statsmodels model summary",
        "path": "results/model_summary.txt",
    },
]


def pipeline_result_context() -> dict[str, Any]:
    plots = []
    for plot in PLOTS:
        path = Path(plot["path"])
        plots.append(
            {
                **plot,
                "exists": path.exists(),
                "url": f"/result-artifacts/{path.relative_to('results')}" if path.exists() else None,
            }
        )

    reports = []
    for report in REPORTS:
        path = Path(report["path"])
        reports.append(
            {
                **report,
                "exists": path.exists(),
                "content": _read_excerpt(path) if path.exists() else "",
            }
        )

    return {
        "metrics": _final_metrics(),
        "charts": _chart_specs(),
        "plots": plots,
        "reports": reports,
        "plots_ready": sum(1 for plot in plots if plot["exists"]),
        "plots_total": len(plots),
        "reports_ready": sum(1 for report in reports if report["exists"]),
        "reports_total": len(reports),
    }


def _chart_specs() -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []

    before_balance = _read_json(Path("results/before_resampling/NFvsF_before_undersampling.json"))
    after_balance = _read_json(Path("results/after_resampling/NFvsF_after_undersampling.json"))
    if before_balance:
        charts.append(_class_balance_chart("class_balance_before", "Class balance before resampling", before_balance))
    if after_balance:
        charts.append(_class_balance_chart("class_balance_after", "Class balance after resampling", after_balance))

    final_metrics = _final_metrics()
    if final_metrics:
        charts.append(
            {
                "id": "final_model_metrics",
                "title": "Final model metrics",
                "type": "bar",
                "source": "results/*.json and classification reports",
                "items": [
                    {
                        "label": f"{metric['split']} {metric['name']}",
                        "value": metric["value"],
                        "display": f"{metric['value']:.3f}",
                    }
                    for metric in final_metrics
                ],
            }
        )
    return charts


def _class_balance_chart(chart_id: str, title: str, payload: dict[str, Any]) -> dict[str, Any]:
    items = []
    total = sum(float(value) for value in payload.values()) or 1
    for label, value in payload.items():
        numeric_value = float(value)
        items.append(
            {
                "label": label.strip(),
                "value": numeric_value,
                "display": f"{int(numeric_value):,}",
                "ratio": numeric_value / total,
            }
        )
    return {
        "id": chart_id,
        "title": title,
        "type": "bar",
        "source": "DVC JSON artifact",
        "items": items,
    }


def _final_metrics() -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for name, split, path in [
        ("accuracy", "balanced_test", Path("results/accuracy_ns.json")),
        ("accuracy", "full_dataset", Path("results/accuracy_wd.json")),
    ]:
        payload = _read_json(path)
        if payload and "model_accuracy" in payload:
            metrics.append(
                {
                    "name": name,
                    "split": split,
                    "value": float(payload["model_accuracy"]),
                    "source": str(path),
                    "ready": True,
                }
            )

    for split, path in [
        ("balanced_test", Path("results/classification_report_ns.txt")),
        ("full_dataset", Path("results/classification_report_wd.txt")),
    ]:
        report_metrics = _parse_fraud_report(path)
        for metric_name, value in report_metrics.items():
            metrics.append(
                {
                    "name": metric_name,
                    "split": split,
                    "value": value,
                    "source": str(path),
                    "ready": True,
                }
            )
    return metrics


def _read_excerpt(path: Path, limit: int = 900) -> str:
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(content) <= limit:
        return content
    return f"{content[:limit].rstrip()}..."


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def _parse_fraud_report(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if parts and parts[0] == "1" and len(parts) >= 4:
            return {
                "fraud_precision": float(parts[1]),
                "fraud_recall": float(parts[2]),
                "fraud_f1": float(parts[3]),
            }
    return {}
