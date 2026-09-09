import argparse
import os
import subprocess
import sys


STAGES = [
    [
        "src/data_understanding.py",
        "results/before_resampling/NFvsF_before_undersampling.json",
        "results/before_resampling/Count_before_undersampling.png",
        "results/before_resampling/distrib_before_undersampling.png",
    ],
    [
        "src/Resampling.py",
        "results/after_resampling/NFvsF_after_undersampling.json",
        "results/after_resampling/Distrib_after_undersampling.png",
        "results/after_resampling/Corr_matrices_comp.png",
    ],
    ["src/prepare.py"],
    [
        "src/train.py",
        "results/finalvar.pkl",
        "results/logit.pkl",
        "results/model_summary.txt",
    ],
    [
        "src/test.py",
        "results/finalvar.pkl",
        "results/logit.pkl",
        "results/accuracy_ns.json",
        "results/confusionmatrix_ns.png",
        "results/classification_report_ns.txt",
        "results/accuracy_wd.json",
        "results/confusionmatrix_wd.png",
        "results/classification_report_wd.txt",
    ],
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local training pipeline stages.")
    parser.add_argument("--version", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip-register", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str((__import__("pathlib").Path(".cache/matplotlib")).resolve()))
    os.makedirs(env["MPLCONFIGDIR"], exist_ok=True)

    for stage in STAGES:
        subprocess.check_call([sys.executable, *stage], env=env)

    if not args.skip_register:
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
