from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the prepared CV Matcher dataset to a Hugging Face dataset repo."
    )
    parser.add_argument(
        "--repo-id",
        default="JeremiahOnu/cv-matcher-data",
        help="Hugging Face dataset repo id, e.g. JeremiahOnu/cv-matcher-data",
    )
    parser.add_argument(
        "--folder",
        default="data",
        help="Local dataset root containing jobs/, candidates/, and README.md",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private if it does not already exist.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Parallel upload workers for large-folder upload.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set. Export your Hugging Face token and try again.")

    folder = Path(args.folder)
    if not folder.exists():
        raise SystemExit(f"Dataset folder does not exist: {folder}")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=folder,
        allow_patterns=["jobs/**", "candidates/**", "README.md"],
        ignore_patterns=["raw/**", "candidates/*_cover.*"],
        num_workers=args.num_workers,
        print_report=True,
        print_report_every=30,
    )

    print(f"Upload complete: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
