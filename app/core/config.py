import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "FraudOps Model Registry"
    registry_path: Path = Path("models/registry.json")
    admin_state_path: Path = Path("models/admin_state.json")
    model_artifacts_dir: Path = Path("models/latest")


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("FRAUD_API_APP_NAME", Settings.app_name),
        registry_path=Path(os.getenv("FRAUD_API_REGISTRY_PATH", Settings.registry_path)),
        admin_state_path=Path(
            os.getenv("FRAUD_API_ADMIN_STATE_PATH", Settings.admin_state_path)
        ),
        model_artifacts_dir=Path(
            os.getenv("FRAUD_API_MODEL_ARTIFACTS_DIR", Settings.model_artifacts_dir)
        ),
    )
