from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ModelStage = Literal["development", "staging", "production", "archived"]
ModelStatus = Literal["candidate", "approved", "degraded", "retired"]


class ModelMetric(BaseModel):
    name: str
    value: float
    split: str
    target_class: Optional[str] = None
    higher_is_better: bool = True


class ModelArtifact(BaseModel):
    name: str
    path: str
    required_for_serving: bool = False
    exists: bool = False


class ModelRisk(BaseModel):
    level: Literal["low", "medium", "high"]
    message: str


class ModelVersion(BaseModel):
    version: str
    name: str
    stage: ModelStage
    status: ModelStatus
    algorithm: str
    trained_at: str
    training_data: str
    git_commit: Optional[str] = None
    source_branch: Optional[str] = None
    params: Dict[str, Any] = {}
    threshold: float = Field(0.5, ge=0, le=1)
    features_count: int
    features: List[str]
    metrics: List[ModelMetric]
    artifacts: List[ModelArtifact]
    risks: List[ModelRisk] = []
    notes: str = ""


class ModelRegistry(BaseModel):
    active_version: Optional[str] = None
    versions: List[ModelVersion]


class ValueDiff(BaseModel):
    before: Any = None
    after: Any = None
    delta: Optional[float] = None
    changed: bool = True


class FeatureDiff(BaseModel):
    added: List[str] = []
    removed: List[str] = []
    unchanged_count: int = 0


class ArtifactDiff(BaseModel):
    name: str
    before_path: Optional[str] = None
    after_path: Optional[str] = None
    before_exists: Optional[bool] = None
    after_exists: Optional[bool] = None
    changed: bool = False


class ModelDiffGate(BaseModel):
    passed: bool
    issues: List[str] = []


class ModelDiff(BaseModel):
    baseline: str
    candidate: str
    metric_diffs: Dict[str, ValueDiff]
    param_diffs: Dict[str, ValueDiff]
    feature_diff: FeatureDiff
    artifact_diffs: List[ArtifactDiff]
    gate: ModelDiffGate


DraftStatus = Literal["draft", "submitted", "training", "completed", "blocked"]


class ParameterDraft(BaseModel):
    id: str
    name: str
    status: DraftStatus = "draft"
    created_at: str
    created_by: str = "admin"
    base_version: Optional[str] = None
    params: Dict[str, Any]
    threshold: float = Field(0.5, ge=0, le=1)
    note: str = ""
    candidate_version: Optional[str] = None


class TrainingJob(BaseModel):
    id: str
    draft_id: str
    status: DraftStatus
    created_at: str
    message: str
    candidate_version: Optional[str] = None


class AdminState(BaseModel):
    drafts: List[ParameterDraft] = []
    jobs: List[TrainingJob] = []
