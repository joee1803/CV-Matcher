from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove candidate cover-letter files locally and optionally from a Hugging Face dataset repo."
    )
    parser.add_argument(
        "--local-folder",
        default="data/candidates",
        help="Local candidates folder containing cand_*_cover.* files.",
    )
    parser.add_argument(
        "--repo-id",
        default="JeremiahOnu/cv-matcher-data",
        help="Hugging Face dataset repo id to clean up.",
    )
    parser.add_argument(
        "--remote-only",
        action="store_true",
        help="Only delete remote cover-letter files.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only delete local cover-letter files.",
    )
    return parser.parse_args()


def remove_local(folder: Path) -> int:
    removed = 0
    for path in folder.glob("cand_*_cover.*"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def remove_remote(repo_id: str) -> int:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set. Export your Hugging Face token and try again.")

    api = HfApi(token=token)
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    delete_patterns = sorted(
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.startswith("candidates/") and "_cover." in sibling.rfilename
    )
    if not delete_patterns:
        return 0

    api.delete_files(
        repo_id=repo_id,
        repo_type="dataset",
        delete_patterns=delete_patterns,
        commit_message="Remove candidate cover-letter files",
    )
    return len(delete_patterns)


def main() -> None:
    args = parse_args()
    if args.local_only and args.remote_only:
        raise SystemExit("Use at most one of --local-only or --remote-only.")

    local_removed = 0
    remote_removed = 0

    if not args.remote_only:
        local_removed = remove_local(Path(args.local_folder))

    if not args.local_only:
        remote_removed = remove_remote(args.repo_id)

    print(f"Local cover letters removed: {local_removed}")
    print(f"Remote cover letters removed: {remote_removed}")


if __name__ == "__main__":
    main()
