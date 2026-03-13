from __future__ import annotations

import re
from collections import Counter


_SECTION_PATTERNS = {
    "summary": ("summary", "profile", "objective", "about"),
    "skills": ("skills", "technical skills", "core skills", "competencies"),
    "experience": ("experience", "work history", "employment", "career"),
    "education": ("education", "academic", "qualification"),
    "certifications": ("certification", "certifications", "licenses"),
    "projects": ("projects", "project experience"),
    "contact": ("contact", "email", "phone", "linkedin"),
}

_STOPWORDS = {
    "and",
    "the",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "has",
    "are",
    "you",
    "your",
    "our",
    "will",
    "years",
    "year",
    "work",
    "team",
    "using",
    "role",
    "job",
    "candidate",
}


# Normalize potential section headings from raw CV lines.
def _normalize_heading(line: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", line.lower()).strip()


# Extract lightweight keyword tokens for overlap scoring.
def _keywords(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\+\#\.-]{2,}", text.lower())
    return [w for w in words if w not in _STOPWORDS]


class MatchExplainer:
    """Lightweight explainability helpers for candidate-job similarity matches."""

    # Split CV content into semantic sections using heading heuristics.
    def extract_sections(self, text: str) -> dict[str, str]:
        sections: dict[str, list[str]] = {"general": []}
        current = "general"
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            heading = _normalize_heading(line)
            switched = False
            for section_name, aliases in _SECTION_PATTERNS.items():
                if any(alias in heading for alias in aliases) and len(heading) <= 40:
                    current = section_name
                    sections.setdefault(current, [])
                    switched = True
                    break
            if switched:
                continue
            sections.setdefault(current, []).append(line)
        return {name: "\n".join(lines).strip() for name, lines in sections.items() if lines}

    # Score section-to-job overlap and keep top matching terms per section.
    def find_matching_phrases(self, cv_sections: dict[str, str], job_text: str) -> list[dict]:
        job_terms = Counter(_keywords(job_text))
        matches: list[dict] = []
        for section_name, section_text in cv_sections.items():
            section_terms = Counter(_keywords(section_text))
            shared = [term for term in section_terms if term in job_terms]
            if not shared:
                continue
            weighted = sorted(
                shared,
                key=lambda term: section_terms[term] + job_terms[term],
                reverse=True,
            )
            top_terms = weighted[:5]
            matches.append(
                {
                    "section": section_name,
                    "score": sum(section_terms[t] + job_terms[t] for t in top_terms),
                    "terms": top_terms,
                }
            )
        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches

    # Build human-readable summary and targeted CV improvement guidance.
    def generate_explanation(
        self,
        matching_phrases: list[dict],
        similarity_score: float,
        missing_terms: list[str] | None = None,
    ) -> str:
        if not matching_phrases:
            missing_text = ""
            if missing_terms:
                missing_text = f" Focus on adding evidence for: {', '.join(missing_terms[:5])}."
            return (
                "Limited direct keyword overlap found. Improve your CV by adding a role-focused summary, "
                "measurable achievements, and the exact tools/skills listed in the job description."
                f"{missing_text}"
            )

        top_sections = [item["section"] for item in matching_phrases[:2]]
        matched_terms: list[str] = []
        for item in matching_phrases:
            for term in item["terms"]:
                if term not in matched_terms:
                    matched_terms.append(term)
                if len(matched_terms) == 6:
                    break
            if len(matched_terms) == 6:
                break

        if similarity_score >= 0.75:
            strength = "Strong alignment"
        elif similarity_score >= 0.55:
            strength = "Moderate alignment"
        else:
            strength = "Partial/Limited alignment"

        improvements: list[str] = []
        if missing_terms:
            improvements.append(f"add clear evidence for {', '.join(missing_terms[:4])}")
        if "experience" not in top_sections:
            improvements.append("expand experience bullets with measurable impact")
        if "skills" not in top_sections:
            improvements.append("add a focused technical skills section")
        if "projects" not in top_sections and similarity_score < 0.75:
            improvements.append("include a relevant project aligned to this role")

        improvement_text = ""
        if improvements:
            improvement_text = f" To improve fit, {', then '.join(improvements)}."

        return (
            f"{strength} driven by {', '.join(top_sections)} content "
            f"(matched terms: {', '.join(matched_terms)})."
            f"{improvement_text}"
        )

    # Full explainability payload consumed by CSV output and UI feedback panels.
    def explain_match(self, cv_text: str, job_text: str, similarity_score: float) -> dict:
        sections = self.extract_sections(cv_text)
        matching_phrases = self.find_matching_phrases(sections, job_text)
        cv_terms = Counter(_keywords(cv_text))
        job_terms = Counter(_keywords(job_text))
        missing_terms = [term for term, _ in job_terms.most_common(20) if term not in cv_terms][:8]
        explanation = self.generate_explanation(matching_phrases, similarity_score, missing_terms=missing_terms)
        return {
            "similarity_score": float(similarity_score),
            "sections_used": [item["section"] for item in matching_phrases[:3]],
            "matched_terms": [term for item in matching_phrases for term in item["terms"]][:10],
            "missing_terms": missing_terms,
            "explanation": explanation,
        }
