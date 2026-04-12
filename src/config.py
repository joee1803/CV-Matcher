"""Project configuration and dataset path helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class Config:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    outputs_dir: Path = Path("outputs")
    rankings_csv: Path = Path("outputs/rankings.csv")
    report_docx: Path = Path("outputs/report.docx")
    report_txt: Path = Path("outputs/report.txt")
    cache_dir: Path = Path("outputs/.cache")
    filters_json: Path = Path("outputs/filters_applied.json")


def _setting(name: str) -> str:
    """Read configuration from env first, then Streamlit secrets when available."""
    value = os.getenv(name, "").strip()
    if value:
        return value

    try:
        import streamlit as st

        secret_value = st.secrets.get(name, "")
        if secret_value is None:
            return ""
        return str(secret_value).strip()
    except Exception:
        return ""


def _local_dataset_paths() -> tuple[Path, Path]:
    return Path("data/jobs"), Path("data/candidates")


def _demo_dataset_paths() -> tuple[Path, Path]:
    return Path("demo_data/jobs"), Path("demo_data/candidates")


@lru_cache(maxsize=1)
def _huggingface_dataset_root() -> Path | None:
    """Download and cache a deployment dataset snapshot from Hugging Face."""
    repo_id = _setting("HF_DATASET_REPO_ID")
    if not repo_id:
        return None

    revision = _setting("HF_DATASET_REVISION") or None
    repo_type = _setting("HF_DATASET_REPO_TYPE") or "dataset"
    token = _setting("HF_TOKEN") or None
    subdir = _setting("HF_DATASET_SUBDIR").strip("/\\")
    allow_patterns = [f"{subdir}/**"] if subdir else None

    snapshot_path = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
            allow_patterns=allow_patterns,
        )
    )
    return snapshot_path / subdir if subdir else snapshot_path


def _huggingface_dataset_paths() -> tuple[Path, Path] | None:
    """Return Hugging Face dataset folders when the deployment env is configured."""
    root = _huggingface_dataset_root()
    if root is None:
        return None
    jobs_dir = root / "jobs"
    candidates_dir = root / "candidates"
    if jobs_dir.exists() and candidates_dir.exists():
        return jobs_dir, candidates_dir
    return None


def dataset_paths() -> tuple[Path, Path]:
    jobs_override = _setting("JOBS_DATA_DIR")
    candidates_override = _setting("CANDIDATES_DATA_DIR")
    if jobs_override and candidates_override:
        return Path(jobs_override), Path(candidates_override)

    data_root = _setting("DATA_ROOT")
    if data_root:
        root = Path(data_root)
        return root / "jobs", root / "candidates"

    primary_jobs, primary_candidates = _local_dataset_paths()
    if primary_jobs.exists() and primary_candidates.exists():
        return primary_jobs, primary_candidates

    hf_paths = _huggingface_dataset_paths()
    if hf_paths is not None:
        return hf_paths

    demo_jobs, demo_candidates = _demo_dataset_paths()
    if demo_jobs.exists() and demo_candidates.exists():
        return demo_jobs, demo_candidates

    return primary_jobs, primary_candidates


def dataset_source() -> str:
    jobs_override = _setting("JOBS_DATA_DIR")
    candidates_override = _setting("CANDIDATES_DATA_DIR")
    if jobs_override and candidates_override:
        return "local"

    data_root = _setting("DATA_ROOT")
    if data_root:
        return "local"

    jobs_dir, _ = dataset_paths()
    normalized = str(jobs_dir).replace("\\", "/").lower()
    if "/demo_data/" in f"/{normalized}/":
        return "demo"
    if "/data/jobs" in normalized or normalized.endswith("data/jobs"):
        return "local"
    return "huggingface"


def mode_paths(mode: str) -> tuple[Path, Path]:
    """Resolve the filesystem roots for each supported run mode."""
    mode = mode.lower().strip()
    if mode == "dataset":
        return dataset_paths()
    if mode == "upload":
        return Path("uploads/jobs"), Path("uploads/candidates")
    raise ValueError("mode must be one of: dataset, upload")
