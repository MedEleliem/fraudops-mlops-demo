from collections.abc import Mapping
from typing import Any, Dict, Iterable, Tuple

from app.ml.schemas import (
    ArtifactDiff,
    FeatureDiff,
    ModelDiff,
    ModelDiffGate,
    ModelMetric,
    ModelVersion,
    ValueDiff,
)


CRITICAL_METRICS = {
    "full_dataset:fraud_recall:1": 0.0,
    "full_dataset:fraud_precision:1": -0.02,
    "full_dataset:fraud_f1:1": -0.02,
}


def compare_model_versions(baseline: ModelVersion, candidate: ModelVersion) -> ModelDiff:
    metric_diffs = _diff_metrics(baseline.metrics, candidate.metrics)
    param_diffs = _diff_flattened_dicts(baseline.params, candidate.params)
    feature_diff = _diff_features(baseline.features, candidate.features)
    artifact_diffs = _diff_artifacts(baseline, candidate)
    gate = _evaluate_gate(metric_diffs, artifact_diffs)

    return ModelDiff(
        baseline=baseline.version,
        candidate=candidate.version,
        metric_diffs=metric_diffs,
        param_diffs=param_diffs,
        feature_diff=feature_diff,
        artifact_diffs=artifact_diffs,
        gate=gate,
    )


def metric_key(metric: ModelMetric) -> str:
    target_class = metric.target_class or "all"
    return f"{metric.split}:{metric.name}:{target_class}"


def _diff_metrics(
    baseline_metrics: Iterable[ModelMetric],
    candidate_metrics: Iterable[ModelMetric],
) -> Dict[str, ValueDiff]:
    baseline_by_key = {metric_key(metric): metric for metric in baseline_metrics}
    candidate_by_key = {metric_key(metric): metric for metric in candidate_metrics}
    keys = sorted(set(baseline_by_key) | set(candidate_by_key))

    diffs: Dict[str, ValueDiff] = {}
    for key in keys:
        before = baseline_by_key.get(key)
        after = candidate_by_key.get(key)
        before_value = before.value if before else None
        after_value = after.value if after else None
        delta = None
        if before_value is not None and after_value is not None:
            delta = after_value - before_value
        diffs[key] = ValueDiff(
            before=before_value,
            after=after_value,
            delta=delta,
            changed=before_value != after_value,
        )
    return diffs


def _diff_flattened_dicts(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Dict[str, ValueDiff]:
    baseline_flat = dict(_flatten_dict(baseline))
    candidate_flat = dict(_flatten_dict(candidate))
    keys = sorted(set(baseline_flat) | set(candidate_flat))

    return {
        key: ValueDiff(
            before=baseline_flat.get(key),
            after=candidate_flat.get(key),
            changed=baseline_flat.get(key) != candidate_flat.get(key),
        )
        for key in keys
        if baseline_flat.get(key) != candidate_flat.get(key)
    }


def _flatten_dict(payload: Mapping[str, Any], prefix: str = "") -> Iterable[Tuple[str, Any]]:
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from _flatten_dict(value, full_key)
        else:
            yield full_key, value


def _diff_features(baseline_features: list[str], candidate_features: list[str]) -> FeatureDiff:
    baseline_set = set(baseline_features)
    candidate_set = set(candidate_features)
    return FeatureDiff(
        added=sorted(candidate_set - baseline_set),
        removed=sorted(baseline_set - candidate_set),
        unchanged_count=len(baseline_set & candidate_set),
    )


def _diff_artifacts(baseline: ModelVersion, candidate: ModelVersion) -> list[ArtifactDiff]:
    baseline_artifacts = {artifact.name: artifact for artifact in baseline.artifacts}
    candidate_artifacts = {artifact.name: artifact for artifact in candidate.artifacts}
    names = sorted(set(baseline_artifacts) | set(candidate_artifacts))

    diffs: list[ArtifactDiff] = []
    for name in names:
        before = baseline_artifacts.get(name)
        after = candidate_artifacts.get(name)
        before_path = before.path if before else None
        after_path = after.path if after else None
        diffs.append(
            ArtifactDiff(
                name=name,
                before_path=before_path,
                after_path=after_path,
                before_exists=before.exists if before else None,
                after_exists=after.exists if after else None,
                changed=before_path != after_path,
            )
        )
    return diffs


def _evaluate_gate(
    metric_diffs: Dict[str, ValueDiff],
    artifact_diffs: list[ArtifactDiff],
) -> ModelDiffGate:
    issues: list[str] = []
    for metric_name, min_delta in CRITICAL_METRICS.items():
        diff = metric_diffs.get(metric_name)
        if diff and diff.delta is not None and diff.delta < min_delta:
            issues.append(
                f"{metric_name} regressed by {diff.delta:.4f}; minimum allowed delta is {min_delta:.4f}."
            )

    missing_serving_artifacts = [
        artifact.name
        for artifact in artifact_diffs
        if artifact.after_exists is False and artifact.name in {"model_pickle", "selected_features_pickle"}
    ]
    if missing_serving_artifacts:
        issues.append(f"Missing serving artifacts: {', '.join(missing_serving_artifacts)}.")

    return ModelDiffGate(passed=not issues, issues=issues)
