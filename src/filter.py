"""Optional pre-ranking filters for candidate documents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_SALARY_PATTERNS = (
    re.compile(r"(?i)(?:salary|expected|expectation|compensation)[^\d]{0,20}(\d[\d,]*k?)"),
    re.compile(r"(?i)(?:\$)\s?(\d[\d,]*k?)"),
)


def _parse_salary_value(raw: str) -> float | None:
    """Parse salary values like `45000` or `45k`."""
    value = raw.strip().lower().replace(",", "")
    if not value:
        return None
    if value.endswith("k"):
        try:
            return float(value[:-1]) * 1000
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def _extract_salary(text: str) -> float | None:
    """Extract the first salary-like value found in candidate text."""
    for pattern in _SALARY_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        parsed = _parse_salary_value(match.group(1))
        if parsed is not None:
            return parsed
    return None


@dataclass
class CandidateFilter:
    """Apply optional skills, location, and salary filters before ranking."""
    required_skills: list[str] = field(default_factory=list)
    location: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None

    @property
    def has_active_filters(self) -> bool:
        return bool(self.required_skills or self.location or self.salary_min is not None or self.salary_max is not None)

    def add_skill_filter(self, skills: list[str]) -> None:
        self.required_skills = [s.strip() for s in skills if s and s.strip()]

    def add_location_filter(self, location: str) -> None:
        cleaned = location.strip()
        self.location = cleaned if cleaned else None

    def add_salary_filter(self, min_val: float | None, max_val: float | None) -> None:
        self.salary_min = min_val
        self.salary_max = max_val

    def reset(self) -> None:
        self.required_skills = []
        self.location = None
        self.salary_min = None
        self.salary_max = None

    def _passes_skills(self, text_lower: str) -> bool:
        if not self.required_skills:
            return True
        return all(skill.lower() in text_lower for skill in self.required_skills)

    def _passes_location(self, text_lower: str) -> bool:
        if not self.location:
            return True
        return self.location.lower() in text_lower

    def _passes_salary(self, text: str) -> tuple[bool, bool]:
        if self.salary_min is None and self.salary_max is None:
            return True, False
        salary = _extract_salary(text)
        if salary is None:
            return False, True
        if self.salary_min is not None and salary < self.salary_min:
            return False, False
        if self.salary_max is not None and salary > self.salary_max:
            return False, False
        return True, False

    def apply(self, candidates: list[dict]) -> tuple[list[dict], dict]:
        """Filter candidates and return both the reduced set and filter stats."""
        if not self.has_active_filters:
            return candidates, {"total_candidates": len(candidates), "filtered_out": 0, "remaining": len(candidates)}

        filtered: list[dict] = []
        filtered_out_by_skills = 0
        filtered_out_by_location = 0
        filtered_out_by_salary = 0
        filtered_out_by_missing_salary = 0

        for candidate in candidates:
            text = str(candidate.get("text", ""))
            text_lower = text.lower()
            if not self._passes_skills(text_lower):
                filtered_out_by_skills += 1
                continue
            if not self._passes_location(text_lower):
                filtered_out_by_location += 1
                continue
            salary_ok, missing_salary = self._passes_salary(text)
            if not salary_ok:
                filtered_out_by_salary += 1
                if missing_salary:
                    filtered_out_by_missing_salary += 1
                continue
            filtered.append(candidate)

        stats = {
            "total_candidates": len(candidates),
            "filtered_out": len(candidates) - len(filtered),
            "remaining": len(filtered),
            "filters": {
                "required_skills": self.required_skills,
                "location": self.location,
                "salary_min": self.salary_min,
                "salary_max": self.salary_max,
                "filtered_out_by_skills": filtered_out_by_skills,
                "filtered_out_by_location": filtered_out_by_location,
                "filtered_out_by_salary": filtered_out_by_salary,
                "filtered_out_by_missing_salary": filtered_out_by_missing_salary,
            },
        }
        return filtered, stats


