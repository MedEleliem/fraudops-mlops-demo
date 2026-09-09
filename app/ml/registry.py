import json
from pathlib import Path

from app.ml.schemas import ModelRegistry, ModelVersion


class RegistryNotFoundError(FileNotFoundError):
    pass


class ModelRegistryStore:
    def __init__(self, registry_path: Path) -> None:
        self.registry_path = registry_path

    def load(self) -> ModelRegistry:
        if not self.registry_path.exists():
            raise RegistryNotFoundError(f"Registry file not found: {self.registry_path}")

        with self.registry_path.open("r", encoding="utf-8") as registry_file:
            payload = json.load(registry_file)

        registry = ModelRegistry.model_validate(payload)
        return self._with_artifact_status(registry)

    def active_model(self) -> ModelVersion:
        registry = self.load()
        if not registry.active_version:
            raise ValueError("No production model has been approved yet")
        return self.get_version(registry.active_version, registry=registry)

    def get_version(
        self,
        version_id: str,
        registry: ModelRegistry | None = None,
    ) -> ModelVersion:
        registry = registry or self.load()
        for version in registry.versions:
            if version.version == version_id:
                return version
        raise ValueError(f"Model version {version_id!r} is not listed")

    def _with_artifact_status(self, registry: ModelRegistry) -> ModelRegistry:
        for version in registry.versions:
            for artifact in version.artifacts:
                artifact.exists = Path(artifact.path).exists()
        return registry
