"""Project configuration and dataset path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    outputs_dir: Path = Path("outputs")
    rankings_csv: Path = Path("outputs/rankings.csv")
    report_docx: Path = Path("outputs/report.docx")
    report_txt: Path = Path("outputs/report.txt")
    cache_dir: Path = Path("outputs/.cache")
    filters_json: Path = Path("outputs/filters_applied.json")


def dataset_paths() -> tuple[Path, Path]:
    """Prefer the local dataset and fall back to the tracked demo dataset."""
    primary_jobs = Path("data/jobs")
    primary_candidates = Path("data/candidates")
    if primary_jobs.exists() and primary_candidates.exists():
        return primary_jobs, primary_candidates

    demo_jobs = Path("demo_data/jobs")
    demo_candidates = Path("demo_data/candidates")
    if demo_jobs.exists() and demo_candidates.exists():
        return demo_jobs, demo_candidates

    return primary_jobs, primary_candidates


def mode_paths(mode: str) -> tuple[Path, Path]:
    """Resolve the filesystem roots for each supported run mode."""
    mode = mode.lower().strip()
    if mode == "dataset":
        return dataset_paths()
    if mode == "upload":
        return Path("uploads/jobs"), Path("uploads/candidates")
    raise ValueError("mode must be one of: dataset, upload")


