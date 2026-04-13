from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repartition flat jobs/candidates folders into subfolders for Hugging Face limits."
    )
    parser.add_argument(
        "--root",
        default="data",
        help="Dataset root containing jobs/ and candidates/.",
    )
    parser.add_argument(
        "--files-per-dir",
        type=int,
        default=5000,
        help="Maximum files to place in each generated subdirectory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned moves without modifying files.",
    )
    return parser.parse_args()


def _target_dir(index: int, files_per_dir: int) -> str:
    return f"{index // files_per_dir:04d}"


def repartition_folder(folder: Path, files_per_dir: int, dry_run: bool) -> int:
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.name != ".gitkeep"
    )
    moved = 0
    for index, path in enumerate(files):
        target_parent = folder / _target_dir(index, files_per_dir)
        target_path = target_parent / path.name
        if path.parent == target_parent:
            continue
        if dry_run:
            print(f"DRY-RUN {path} -> {target_path}")
        else:
            target_parent.mkdir(parents=True, exist_ok=True)
            path.replace(target_path)
        moved += 1
    return moved


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    jobs_dir = root / "jobs"
    candidates_dir = root / "candidates"
    if not jobs_dir.exists() or not candidates_dir.exists():
        raise SystemExit("Expected dataset root with jobs/ and candidates/ folders.")

    moved_jobs = repartition_folder(jobs_dir, args.files_per_dir, args.dry_run)
    moved_candidates = repartition_folder(candidates_dir, args.files_per_dir, args.dry_run)
    print(f"Jobs moved: {moved_jobs}")
    print(f"Candidates moved: {moved_candidates}")


if __name__ == "__main__":
    main()
