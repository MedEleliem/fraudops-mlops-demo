import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DVC retraining, register artifacts, and log MLflow results.")
    parser.add_argument("--version", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip-dvc", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    env.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    env.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
    env.setdefault("DVC_SITE_CACHE_DIR", str(Path(".cache/dvc/site").resolve()))
    env.setdefault("DVC_GLOBAL_CONFIG_DIR", str(Path(".config/dvc").resolve()))
    env.setdefault("DVC_SYSTEM_CONFIG_DIR", str(Path(".config/dvc-system").resolve()))
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    for path_key in ("MPLCONFIGDIR", "DVC_SITE_CACHE_DIR", "DVC_GLOBAL_CONFIG_DIR", "DVC_SYSTEM_CONFIG_DIR"):
        Path(env[path_key]).mkdir(parents=True, exist_ok=True)

    if not args.skip_dvc:
        subprocess.check_call([sys.executable, "-m", "dvc", "repro"], env=env)

    subprocess.check_call(
        [
            sys.executable,
            "scripts/register_model.py",
            "--version",
            args.version,
            "--copy-artifacts",
            "--stage",
            "development",
            "--status",
            "candidate",
            "--threshold",
            str(args.threshold),
        ],
        env=env,
    )
    subprocess.check_call(
        [
            sys.executable,
            "scripts/log_mlflow_run.py",
            "--run-name",
            args.version,
        ],
        env=env,
    )


if __name__ == "__main__":
    main()
