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


def mode_paths(mode: str) -> tuple[Path, Path]:
    mode = mode.lower().strip()
    if mode == "dataset":
        return Path("data/jobs"), Path("data/candidates")
    if mode == "upload":
        return Path("uploads/jobs"), Path("uploads/candidates")
    raise ValueError("mode must be one of: dataset, upload")
