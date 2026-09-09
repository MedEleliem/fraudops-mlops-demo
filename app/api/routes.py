import os

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.config import Settings, get_settings
from app.ml.admin import AdminStateStore
from app.ml.dashboard import dashboard_context, mlflow_run_summary
from app.ml.diff import compare_model_versions
from app.ml.pipeline_results import pipeline_result_context
from app.ml.registry import ModelRegistryStore
from app.ml.schemas import AdminState, ModelDiff, ModelRegistry, ModelVersion, ParameterDraft

templates = Jinja2Templates(directory="app/templates")
router = APIRouter()


def get_registry_store(settings: Settings = Depends(get_settings)) -> ModelRegistryStore:
    return ModelRegistryStore(settings.registry_path)


def get_admin_store(
    settings: Settings = Depends(get_settings),
    registry_store: ModelRegistryStore = Depends(get_registry_store),
) -> AdminStateStore:
    return AdminStateStore(settings.admin_state_path, registry_store)


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "model-registry"}


@router.get("/api/models", response_model=ModelRegistry)
def list_models(store: ModelRegistryStore = Depends(get_registry_store)) -> ModelRegistry:
    return store.load()


@router.get("/api/models/active", response_model=ModelVersion)
def active_model(store: ModelRegistryStore = Depends(get_registry_store)) -> ModelVersion:
    try:
        return store.active_model()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/models/{version_id}", response_model=ModelVersion)
def model_version(
    version_id: str,
    store: ModelRegistryStore = Depends(get_registry_store),
) -> ModelVersion:
    try:
        return store.get_version(version_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/models/{candidate_id}/diff/{baseline_id}", response_model=ModelDiff)
def model_diff(
    candidate_id: str,
    baseline_id: str,
    store: ModelRegistryStore = Depends(get_registry_store),
) -> ModelDiff:
    registry = store.load()
    if not registry.versions:
        raise HTTPException(status_code=404, detail="No model versions are registered yet")
    try:
        baseline = store.get_version(baseline_id, registry=registry)
        candidate = store.get_version(candidate_id, registry=registry)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return compare_model_versions(baseline=baseline, candidate=candidate)


@router.get("/api/models/latest-submitted/diff", response_model=ModelDiff)
def latest_submitted_diff(
    store: ModelRegistryStore = Depends(get_registry_store),
    admin_store: AdminStateStore = Depends(get_admin_store),
) -> ModelDiff:
    registry = store.load()
    if not registry.versions:
        raise HTTPException(status_code=404, detail="No model versions are registered yet")
    submitted_versions = [
        draft.candidate_version
        for draft in admin_store.load().drafts
        if draft.candidate_version
    ]
    registered = {version.version for version in registry.versions}
    submitted_versions = [version for version in submitted_versions if version in registered]
    if not submitted_versions:
        active = store.active_model()
        return compare_model_versions(baseline=active, candidate=active)

    candidate = store.get_version(submitted_versions[0], registry=registry)
    baseline_id = submitted_versions[1] if len(submitted_versions) > 1 else registry.active_version
    baseline = store.get_version(baseline_id, registry=registry)
    return compare_model_versions(baseline=baseline, candidate=candidate)


@router.get("/api/admin/state", response_model=AdminState)
def admin_state(store: AdminStateStore = Depends(get_admin_store)) -> AdminState:
    return store.load()


@router.get("/api/admin/runtime")
def admin_runtime() -> dict:
    real_retraining = os.getenv("FRAUD_API_ENABLE_RETRAINING", "false").lower() == "true"
    return {
        "real_retraining_enabled": real_retraining,
        "mode": "training" if real_retraining else "portfolio_demo",
    }


@router.get("/api/pipeline/results")
def pipeline_results() -> dict:
    return pipeline_result_context()


@router.get("/api/mlflow/runs/{version_id}")
def mlflow_run(version_id: str) -> dict:
    return mlflow_run_summary(version_id)


@router.post("/api/admin/drafts", response_model=ParameterDraft)
async def create_parameter_draft(
    request: Request,
    store: AdminStateStore = Depends(get_admin_store),
) -> ParameterDraft:
    payload = await request.json()
    try:
        return store.create_draft(payload)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Missing field: {exc}") from exc


@router.post("/api/admin/drafts/{draft_id}/submit")
def submit_parameter_draft(
    draft_id: str,
    store: AdminStateStore = Depends(get_admin_store),
) -> dict:
    try:
        run_pipeline = os.getenv("FRAUD_API_ENABLE_RETRAINING", "false").lower() == "true"
        draft, job, candidate = store.submit_retraining(draft_id, run_pipeline=run_pipeline)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "draft": draft,
        "job": job,
        "candidate": candidate,
    }


@router.post("/api/admin/models/{version_id}/approve", response_model=ModelVersion)
def approve_model_for_serving(
    version_id: str,
    store: AdminStateStore = Depends(get_admin_store),
) -> ModelVersion:
    try:
        return store.approve_model(version_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/api/admin/models/{version_id}/retrain")
def retrain_from_snapshot(
    version_id: str,
    store: AdminStateStore = Depends(get_admin_store),
    registry_store: ModelRegistryStore = Depends(get_registry_store),
) -> dict:
    """Create a traceable retraining run from an existing dated snapshot."""
    try:
        source = registry_store.get_version(version_id)
        draft = store.create_draft(
            {
                "name": f"Rerun of {source.name}",
                "base_version": source.version,
                "params": source.params,
                "threshold": source.threshold,
                "note": f"Reproduced from snapshot {source.version} with the same parameters.",
            }
        )
        run_pipeline = os.getenv("FRAUD_API_ENABLE_RETRAINING", "false").lower() == "true"
        draft, job, candidate = store.submit_retraining(draft.id, run_pipeline=run_pipeline)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"draft": draft, "job": job, "candidate": candidate}


@router.get("/project", response_class=HTMLResponse)
@router.get("/presentation", response_class=HTMLResponse)
def project_story(request: Request, settings: Settings = Depends(get_settings)) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "project.html",
        {"app_name": settings.app_name},
    )


def _selection_context(
    request: Request,
    settings: Settings,
    store: ModelRegistryStore,
    admin_store: AdminStateStore,
) -> dict:
    registry = store.load()
    if not registry.versions:
        raise ValueError("No model versions are registered yet")

    admin = admin_store.load()
    registered = {version.version for version in registry.versions}
    reference = (
        store.get_version(registry.active_version, registry=registry)
        if registry.active_version
        else registry.versions[0]
    )
    submitted_versions = [
        draft.candidate_version
        for draft in admin.drafts
        if draft.candidate_version and draft.candidate_version in registered
    ]
    candidate_id = submitted_versions[0] if submitted_versions else reference.version
    selected_id = request.query_params.get("model") or candidate_id
    if selected_id not in registered:
        selected_id = candidate_id

    requested_baseline = request.query_params.get("baseline")
    fallback_baseline = next(
        (version.version for version in registry.versions if version.version != selected_id),
        reference.version,
    )
    comparison_baseline_id = (
        requested_baseline if requested_baseline in registered else fallback_baseline
    )
    candidate = store.get_version(selected_id, registry=registry)
    comparison_baseline = store.get_version(comparison_baseline_id, registry=registry)
    diff = compare_model_versions(baseline=comparison_baseline, candidate=candidate)
    preview_metrics = {
        metric.name: metric.value
        for metric in reference.metrics
        if metric.split == "full_dataset"
    }
    drafts_by_id = {draft.id: draft for draft in admin.drafts}
    models_by_id = {version.version: version for version in registry.versions}
    run_history = []
    for job in admin.jobs:
        draft = drafts_by_id.get(job.draft_id)
        model = models_by_id.get(job.candidate_version or "")
        run_history.append(
            {
                "job": job,
                "draft": draft,
                "model": model,
                "mlflow": mlflow_run_summary(model.version) if model else None,
            }
        )

    return {
        "app_name": settings.app_name,
        "registry": registry,
        "active": reference,
        "has_production": bool(registry.active_version),
        "candidate": candidate,
        "comparison_baseline": comparison_baseline,
        "diff": diff,
        "dashboard": dashboard_context(registry, candidate, comparison_baseline, diff),
        "admin": admin,
        "run_history": run_history,
        "preview_metrics": preview_metrics,
        "pipeline_results": pipeline_result_context(),
    }


@router.get("/runs", response_class=HTMLResponse)
def runs_page(
    request: Request,
    settings: Settings = Depends(get_settings),
    store: ModelRegistryStore = Depends(get_registry_store),
    admin_store: AdminStateStore = Depends(get_admin_store),
) -> HTMLResponse:
    registry = store.load()
    if not registry.versions:
        return templates.TemplateResponse(
            request,
            "runs.html",
            {"app_name": settings.app_name, "registry": registry, "admin": admin_store.load()},
        )
    return templates.TemplateResponse(
        request,
        "runs.html",
        _selection_context(request, settings, store, admin_store),
    )


@router.get("/", response_class=HTMLResponse)
def registry_page(
    request: Request,
    settings: Settings = Depends(get_settings),
    store: ModelRegistryStore = Depends(get_registry_store),
    admin_store: AdminStateStore = Depends(get_admin_store),
) -> HTMLResponse:
    registry = store.load()
    if not registry.versions:
        return templates.TemplateResponse(
            request,
            "empty_registry.html",
            {
                "app_name": settings.app_name,
                "registry": registry,
                "admin": admin_store.load(),
                "pipeline_results": pipeline_result_context(),
                "default_params": {
                    "threshold": 0.5,
                    "p_value": 0.05,
                    "split_param": 0.2,
                    "percentage": 1,
                },
            },
        )
    return templates.TemplateResponse(
        request,
        "registry.html",
        _selection_context(request, settings, store, admin_store),
    )
