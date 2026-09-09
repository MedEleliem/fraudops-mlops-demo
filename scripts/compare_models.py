import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.ml.diff import compare_model_versions
from app.ml.registry import ModelRegistryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two registered model versions.")
    parser.add_argument("candidate")
    parser.add_argument("baseline")
    parser.add_argument("--registry", type=Path, default=Path("models/registry.json"))
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    store = ModelRegistryStore(args.registry)
    registry = store.load()
    baseline = store.get_version(args.baseline, registry=registry)
    candidate = store.get_version(args.candidate, registry=registry)
    diff = compare_model_versions(baseline=baseline, candidate=candidate)

    print(json.dumps(diff.model_dump(), indent=2))
    if args.fail_on_gate and not diff.gate.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
