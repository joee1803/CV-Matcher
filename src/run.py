from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from time import perf_counter
from typing import Any

import pandas as pd

from .cache import EmbeddingCache
from .config import Config, mode_paths
from .embed import Embedder
from .explain import MatchExplainer
from .filter import CandidateFilter
from .io import CandidateDoc, ensure_dir, load_candidate_docs, load_jobs
from .match import cosine_similarity_matrix, rank_candidates
from .report import write_report_docx, write_report_txt


# Merge CV/cover files into one record per candidate for embedding + ranking.
def combine_candidate_docs(docs: list[CandidateDoc], doc_mode: str = "cv_only") -> list[dict[str, str]]:
    grouped: dict[str, dict[str, str]] = {}
    for d in docs:
        grouped.setdefault(d.candidate_id, {})
        grouped[d.candidate_id][d.kind] = d.text

    combined: list[dict[str, str]] = []
    for candidate_id, parts in sorted(grouped.items()):
        if "cv" not in parts:
            continue
        cv_text = parts["cv"].strip()
        cover_text = parts.get("cover", "").strip()
        if doc_mode == "cv_and_cover" and cover_text:
            text = f"{cv_text}\n\n{cover_text}".strip()
        else:
            text = cv_text
        if text:
            combined.append(
                {
                    "candidate_id": candidate_id,
                    "cv_text": cv_text,
                    "cover_text": cover_text,
                    "text": text,
                }
            )
    return combined


# Prefer explicit title from job text; otherwise fallback to job id.
def _extract_job_title(text: str, doc_id: str) -> str:
    first_line = (text.splitlines()[0].strip() if text else "")
    m = re.match(r"^JOB TITLE:\s*(.+)$", first_line, flags=re.IGNORECASE)
    if m:
        title = m.group(1).strip()
        if title:
            return title
    return doc_id


# CLI contract for dataset/upload matching.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Candidate/job semantic matching")
    parser.add_argument("--mode", default="dataset", choices=["dataset"])
    parser.add_argument("--candidate-doc-mode", default="cv_only", choices=["cv_only", "cv_and_cover"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--write-text-report", action="store_true")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--require-skills", nargs="*")
    parser.add_argument("--location")
    parser.add_argument("--salary-min", type=float)
    parser.add_argument("--salary-max", type=float)
    parser.add_argument("--output-explanations", dest="output_explanations", action="store_true", default=True)
    parser.add_argument("--no-output-explanations", dest="output_explanations", action="store_false")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--refresh-online-data", action="store_true")
    parser.add_argument("--refresh-max-candidates", type=int, default=500)
    parser.add_argument("--refresh-max-jobs", type=int, default=100)
    parser.add_argument("--refresh-seed", type=int, default=42)
    parser.add_argument("--refresh-no-force", action="store_true")
    parser.add_argument("--use-faiss", action="store_true", help="Use FAISS/Annoy index if available")
    parser.add_argument(
        "--explanation-top-n-jobs",
        type=int,
        default=0,
        help="If > 0, generate explanations only for the top-N jobs by best similarity score.",
    )
    return parser.parse_args()


# Optional online refresh path used by CLI and Streamlit dataset mode.
def refresh_dataset_from_online(
    max_candidates: int,
    max_jobs: int,
    seed: int,
    force: bool,
) -> dict[str, str | int]:
    cmd = [
        sys.executable,
        "-m",
        "src.prepare_datasets",
        "--download",
        "--max-candidates",
        str(max_candidates),
        "--max-jobs",
        str(max_jobs),
        "--seed",
        str(seed),
        "--target-mode",
        "dataset",
    ]
    if force:
        cmd.append("--force")

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return {
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


# End-to-end matcher pipeline used by both CLI and UI entrypoints.
def run_matching(
    mode: str,
    candidate_doc_mode: str,
    top_k: int,
    write_text_report: bool,
    use_cache: bool,
    clear_cache: bool,
    required_skills: list[str] | None,
    location: str | None,
    salary_min: float | None,
    salary_max: float | None,
    output_explanations: bool = True,
    allow_model_download: bool = False,
    refresh_online_data: bool = False,
    refresh_max_candidates: int = 500,
    refresh_max_jobs: int = 100,
    refresh_seed: int = 42,
    refresh_force: bool = True,
    explanation_top_n_jobs: int = 0,
    jobs_override: list | None = None,
    candidates_override: list[dict[str, str]] | None = None,
    write_outputs: bool = True,
    use_faiss: bool = False,
    max_jobs: int | None = None,
    max_candidates: int | None = None,
    sample_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if mode != "dataset":
        raise RuntimeError("Upload mode has been removed. Use --mode dataset.")

    cfg = Config()
    ensure_dir(cfg.outputs_dir)

    # 1) Load and optionally refresh datasets.
    t0 = perf_counter()
    refresh_info: dict[str, str | int] | None = None
    if jobs_override is None or candidates_override is None:
        if mode == "dataset" and refresh_online_data:
            refresh_info = refresh_dataset_from_online(
                max_candidates=refresh_max_candidates,
                max_jobs=refresh_max_jobs,
                seed=refresh_seed,
                force=refresh_force,
            )
        jobs_dir, candidates_dir = mode_paths(mode)
        jobs = load_jobs(jobs_dir, max_items=max_jobs, seed=sample_seed)
        candidate_docs = load_candidate_docs(candidates_dir, max_items=max_candidates, seed=sample_seed)
        candidates = combine_candidate_docs(candidate_docs, doc_mode=candidate_doc_mode)
    else:
        jobs = jobs_override
        candidates = candidates_override
    t_load = perf_counter() - t0

    if not jobs:
        raise RuntimeError("No supported job files found.")
    if not candidates:
        raise RuntimeError("No candidate CVs found. Expected cand_XXXX_cv.*")

    # 2) Apply candidate filters before embedding to reduce cost.
    filter_obj = CandidateFilter()
    if required_skills:
        filter_obj.add_skill_filter(required_skills)
    if location:
        filter_obj.add_location_filter(location)
    if salary_min is not None or salary_max is not None:
        filter_obj.add_salary_filter(salary_min, salary_max)

    t1 = perf_counter()
    candidates, filter_stats = filter_obj.apply(candidates)
    t_filter = perf_counter() - t1
    if not candidates:
        raise RuntimeError("All candidates were filtered out. Relax filter constraints and try again.")

    # 3) Embed jobs/candidates with optional persistent cache.
    t2 = perf_counter()
    embedder = Embedder(cfg.model_name, allow_download=allow_model_download)
    cache = EmbeddingCache(cfg.cache_dir)
    if clear_cache:
        cache.clear()
    if use_cache:
        job_emb = cache.encode_batch([j.text for j in jobs], embedder)
        cand_emb = cache.encode_batch([c["text"] for c in candidates], embedder)
    else:
        job_emb = embedder.encode([j.text for j in jobs], batch_size=64)
        cand_emb = embedder.encode([c["text"] for c in candidates], batch_size=64)
    t_embed = perf_counter() - t2

    # 4) Rank candidates for each job (index-backed or direct cosine).
    t3 = perf_counter()
    rows: list[dict[str, Any]] = []
    cand_ids = [c["candidate_id"] for c in candidates]

    if use_faiss:
        try:
            from .indexer import build_or_load_index, query_index

            index_type, index_obj, stored_emb = build_or_load_index(cand_emb, cfg.outputs_dir, prefer_faiss=True)
            if index_type is not None:
                idxs, sims = query_index(index_type, index_obj, job_emb, top_k, stored_emb)
                for job_idx, job in enumerate(jobs):
                    for rank_pos, cand_pos in enumerate(idxs[job_idx]):
                        if int(cand_pos) < 0:
                            break
                        candidate_id = cand_ids[int(cand_pos)]
                        rows.append(
                            {
                                "job_id": job.doc_id,
                                "job_title": _extract_job_title(job.text, job.doc_id),
                                "rank": rank_pos + 1,
                                "candidate_id": candidate_id,
                                "similarity": round(float(sims[job_idx][rank_pos]), 6),
                                "explanation": "",
                            }
                        )
        except Exception:
            rows = []

    if not rows:
        chunk_size = 32
        for start in range(0, len(jobs), chunk_size):
            sims_chunk = cosine_similarity_matrix(job_emb[start : start + chunk_size], cand_emb)
            for rel_i, sim_scores in enumerate(sims_chunk):
                job_idx = start + rel_i
                ranked = rank_candidates(sim_scores, cand_ids, top_k)
                job = jobs[job_idx]
                for rank, (candidate_id, similarity) in enumerate(ranked, start=1):
                    rows.append(
                        {
                            "job_id": job.doc_id,
                            "job_title": _extract_job_title(job.text, job.doc_id),
                            "rank": rank,
                            "candidate_id": candidate_id,
                            "similarity": round(float(similarity), 6),
                            "explanation": "",
                        }
                    )
    t_match = perf_counter() - t3

    # 5) Generate explanations (full or top-N jobs in fast mode).
    t4 = perf_counter()
    explained_rows = 0
    if output_explanations:
        explainer = MatchExplainer()
        cand_by_id = {c["candidate_id"]: c for c in candidates}
        job_by_id = {j.doc_id: j for j in jobs}
        job_scope: set[str] | None = None
        if explanation_top_n_jobs and explanation_top_n_jobs > 0:
            best_by_job: dict[str, float] = {}
            for row in rows:
                jid = str(row["job_id"])
                score = float(row["similarity"])
                prev = best_by_job.get(jid)
                if prev is None or score > prev:
                    best_by_job[jid] = score
            top_jobs = sorted(best_by_job.items(), key=lambda kv: kv[1], reverse=True)[:explanation_top_n_jobs]
            job_scope = {job_id for job_id, _ in top_jobs}
        for row in rows:
            if job_scope is not None and str(row["job_id"]) not in job_scope:
                continue
            cand = cand_by_id[row["candidate_id"]]
            job = job_by_id.get(row["job_id"])
            if job is None:
                continue
            cv_text = cand.get("cv_text") or cand.get("text", "")
            explained = explainer.explain_match(cv_text, job.text, row["similarity"])
            row["explanation"] = explained.get("explanation", "")
            explained_rows += 1
    t_explain = perf_counter() - t4

    df = pd.DataFrame(rows, columns=["job_id", "job_title", "rank", "candidate_id", "similarity", "explanation"])

    # 6) Persist outputs and runtime metadata.
    if write_outputs:
        df.to_csv(cfg.rankings_csv, index=False)
        write_report_docx(cfg.report_docx, mode, top_k, rows)
        if write_text_report:
            write_report_txt(cfg.report_txt, mode, top_k, rows)
        if filter_obj.has_active_filters:
            cfg.filters_json.write_text(json.dumps(filter_stats, indent=2), encoding="utf-8")

    metrics: dict[str, Any] = {
        "jobs_loaded": len(jobs),
        "candidates_ranked": len(candidates),
        "rows": len(rows),
        "timings": {
            "load_seconds": round(t_load, 3),
            "filter_seconds": round(t_filter, 3),
            "embed_seconds": round(t_embed, 3),
            "match_seconds": round(t_match, 3),
            "explain_seconds": round(t_explain, 3),
            "total_seconds": round(t_load + t_filter + t_embed + t_match + t_explain, 3),
        },
        "cache_stats": cache.stats() if use_cache else None,
        "filter_stats": filter_stats if filter_obj.has_active_filters else None,
        "refresh_info": refresh_info,
        "explained_rows": explained_rows,
        "sampling": {
            "max_jobs": max_jobs,
            "max_candidates": max_candidates,
            "sample_seed": sample_seed,
        },
    }
    return df, metrics


# CLI entrypoint wrapper around run_matching.
def main() -> None:
    args = parse_args()
    _, metrics = run_matching(
        mode=args.mode,
        candidate_doc_mode=args.candidate_doc_mode,
        top_k=args.top_k,
        write_text_report=args.write_text_report,
        use_cache=args.use_cache,
        clear_cache=args.clear_cache,
        required_skills=args.require_skills,
        location=args.location,
        salary_min=args.salary_min,
        salary_max=args.salary_max,
        output_explanations=args.output_explanations,
        allow_model_download=args.allow_model_download,
        refresh_online_data=args.refresh_online_data,
        refresh_max_candidates=args.refresh_max_candidates,
        refresh_max_jobs=args.refresh_max_jobs,
        refresh_seed=args.refresh_seed,
        refresh_force=not args.refresh_no_force,
        explanation_top_n_jobs=args.explanation_top_n_jobs,
        write_outputs=True,
        use_faiss=args.use_faiss,
        max_jobs=args.max_jobs,
        max_candidates=args.max_candidates,
        sample_seed=args.sample_seed,
    )

    cfg = Config()
    print(f"Jobs loaded: {metrics['jobs_loaded']}")
    print(f"Candidates ranked: {metrics['candidates_ranked']}")
    print(f"Rankings rows: {metrics['rows']}")
    print(f"Saved: {cfg.rankings_csv}")
    print(f"Saved: {cfg.report_docx}")
    if args.write_text_report:
        print(f"Saved: {cfg.report_txt}")
    if metrics["filter_stats"]:
        print(f"Saved: {cfg.filters_json}")
    if metrics["cache_stats"]:
        c = metrics["cache_stats"]
        print(f"Cache hits/misses: {c['hits']}/{c['misses']} (hit_rate={c['hit_rate']:.2%})")
    if metrics["refresh_info"]:
        print("Online refresh: done")
    if args.output_explanations:
        print(f"Explained rows: {metrics.get('explained_rows', 0)}")
    t = metrics["timings"]
    print(
        "Runtime (s): "
        f"load={t['load_seconds']} filter={t['filter_seconds']} embed={t['embed_seconds']} "
        f"match={t['match_seconds']} explain={t['explain_seconds']} total={t['total_seconds']}"
    )


if __name__ == "__main__":
    main()
