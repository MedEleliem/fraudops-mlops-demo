import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the versioned client serving Docker image.")
    parser.add_argument("--version", required=True)
    parser.add_argument("--registry", default="")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    production_model = Path("models/production/logit.pkl")
    production_features = Path("models/production/finalvar.pkl")
    if not production_model.exists() or not production_features.exists():
        raise SystemExit("No approved production model found in models/production.")

    image = f"fraud-serving:{args.version}"
    if args.registry:
        image = f"{args.registry.rstrip('/')}/{image}"

    subprocess.check_call(["docker", "build", "-f", "Dockerfile.serving", "-t", image, "."])
    if args.push:
        subprocess.check_call(["docker", "push", image])
    print(image)


if __name__ == "__main__":
    main()
