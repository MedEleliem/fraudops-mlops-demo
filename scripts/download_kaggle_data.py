import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path


DATASET = "mlg-ulb/creditcardfraud"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Kaggle credit card fraud dataset.")
    parser.add_argument("--output", type=Path, default=Path("creditcard.csv"))
    parser.add_argument("--dataset", default=DATASET)
    args = parser.parse_args()

    if args.output.exists():
        print(f"{args.output} already exists")
        return

    if shutil.which("kaggle"):
        _download_with_kaggle_cli(args.dataset, args.output)
        return

    cache_dir = Path(".kaggle-data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KAGGLEHUB_CACHE", str(cache_dir.resolve()))

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "Install Kaggle tooling first: pip install kagglehub or pip install kaggle. "
            "Kaggle may require credentials in ~/.kaggle/kaggle.json."
        ) from exc

    dataset_path = Path(kagglehub.dataset_download(args.dataset))
    candidates = list(dataset_path.rglob("creditcard.csv"))
    if not candidates:
        raise SystemExit(f"creditcard.csv not found in downloaded dataset: {dataset_path}")
    shutil.copy2(candidates[0], args.output)
    print(args.output)


def _download_with_kaggle_cli(dataset: str, output: Path) -> None:
    download_dir = Path(".kaggle-data")
    download_dir.mkdir(exist_ok=True)
    subprocess.check_call(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset,
            "-p",
            str(download_dir),
        ]
    )
    zip_files = sorted(download_dir.glob("*.zip"))
    if not zip_files:
        raise SystemExit("Kaggle download did not produce a zip file.")
    with zipfile.ZipFile(zip_files[-1]) as archive:
        archive.extract("creditcard.csv", download_dir)
    shutil.copy2(download_dir / "creditcard.csv", output)
    print(output)


if __name__ == "__main__":
    main()
