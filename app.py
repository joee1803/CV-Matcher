from __future__ import annotations

import re
import random
from pathlib import Path

import pandas as pd
import streamlit as st

from src.io import count_dataset_items, read_document_cached
from src.run import run_matching

MAX_UI_ROWS = 5000
DEFAULT_SUBSET_JOBS = 60
BASE_SUBSET_SEED = 42
USE_SUBSET_MODE = True
USE_EMBEDDING_CACHE = True
GENERATE_EXPLANATIONS = True


# UI theme + accessibility styles for Streamlit components.
def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Source+Sans+3:wght@400;600;700&display=swap');
        :root {
          --bg: radial-gradient(circle at 10% 10%, #f7fbff 0%, #edf7f3 42%, #fff7eb 100%);
          --ink: #0f172a;
          --muted: #334155;
          --accent: #0f766e;
          --accent-2: #d97706;
          --card: rgba(255,255,255,0.9);
          --border: #dbe7f3;
          --surface: #f8fcff;
        }
        .stApp { background: var(--bg); color: var(--ink); font-family: "Source Sans 3", "Segoe UI", sans-serif; }
        [data-testid="stAppViewContainer"] { background: var(--bg) !important; }
        [data-testid="stHeader"] { background: #ffffff !important; border-bottom: 1px solid var(--border); }
        [data-testid="stToolbar"] { right: 0.5rem; }
        [data-testid="stSidebar"] { background: #f8fafc !important; border-right: 1px solid var(--border); }
        [data-testid="stSidebar"] > div:first-child { background: #f8fafc !important; }
        .stApp, .stMarkdown, .stText, .stCaption, .st-emotion-cache-10trblm, .st-emotion-cache-q8sbsg { color: var(--ink) !important; }
        [data-testid="stSidebar"], [data-testid="stSidebar"] * { color: var(--ink) !important; fill: var(--ink) !important; }
        [data-testid="stSidebar"] [data-testid="stNumberInput"] input,
        [data-testid="stSidebar"] [data-testid="stNumberInput"] button,
        [data-testid="stSidebar"] [data-testid="stNumberInputContainer"] input,
        [data-testid="stSidebar"] [data-testid="stNumberInputContainer"] button {
          background: #ffffff !important;
          color: var(--ink) !important;
          border-color: var(--border) !important;
        }
        [data-testid="stSidebar"] input[type="number"] {
          background: #ffffff !important;
          color: var(--ink) !important;
          caret-color: var(--ink) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="slider"] div { color: var(--ink) !important; }
        [data-testid="stSidebar"] [data-baseweb="select"], [data-testid="stSidebar"] [data-baseweb="input"] {
          background: #ffffff !important;
          color: var(--ink) !important;
        }
        [data-testid="stCheckbox"] label, [data-testid="stRadio"] label { color: var(--ink) !important; }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: var(--ink) !important; }
        [data-testid="stExpander"] details {
          background: #ffffff !important;
          border: 1px solid var(--border) !important;
          border-radius: 8px;
        }
        [data-testid="stInfo"], [data-testid="stSuccess"], [data-testid="stWarning"], [data-testid="stError"] { color: var(--ink) !important; }
        [data-testid="stDataFrame"] {
          background: #ffffff !important;
          border: 1px solid var(--border) !important;
          border-radius: 10px !important;
          overflow: hidden !important;
        }
        [data-testid="stDataFrame"] * { color: #0f172a !important; }
        h1, h2, h3 { color: var(--ink); letter-spacing: 0.2px; font-family: "Space Grotesk", "Segoe UI", sans-serif; }
        .hero {
          border: 1px solid var(--border);
          border-radius: 18px;
          padding: 18px 20px;
          background: linear-gradient(120deg, #ffffff 0%, #f6fffd 100%);
          box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
          margin-bottom: 12px;
        }
        .chip {
          display:inline-block;
          padding: 4px 10px;
          border-radius: 999px;
          border: 1px solid var(--border);
          background: #ffffff;
          margin-right: 6px;
          font-size: 12px;
          color: var(--muted);
        }
        .snapshot {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 14px;
          padding: 12px 14px;
          margin-bottom: 10px;
        }
        .snapshot-label {
          color: var(--muted);
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.4px;
        }
        .snapshot-value {
          color: var(--ink);
          font-size: 24px;
          font-family: "Space Grotesk", "Segoe UI", sans-serif;
          font-weight: 700;
          line-height: 1.2;
        }
        .stButton > button {
          background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%);
          border: none;
          color: white;
          font-weight: 700;
          border-radius: 10px;
          box-shadow: 0 8px 18px rgba(15, 118, 110, 0.22);
        }
        .stDownloadButton > button {
          background: #ffffff !important;
          color: var(--ink) !important;
          border: 1px solid var(--border) !important;
          font-weight: 600;
        }
        .stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }
        .run-mode-banner {
          background: #e8f3ff;
          color: #0b3a66;
          border: 1px solid #c9dff5;
          border-radius: 10px;
          padding: 10px 12px;
          margin: 8px 0 12px 0;
          font-weight: 600;
        }
        .tips {
          background: #f0f9ff;
          border: 1px solid #cde8fb;
          color: #0b3a66;
          border-radius: 12px;
          padding: 10px 12px;
          margin: 8px 0 10px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _candidate_label(candidate_id: str) -> str:
    return re.sub(r"^upload_cand_(\d+)$", lambda m: f"Candidate {int(m.group(1))}", candidate_id)


@st.cache_data(show_spinner=False)
def _current_dataset_limits() -> tuple[int, int]:
    jobs_count = count_dataset_items(Path("data/jobs"), r"^jd_\d+$")
    cv_count = count_dataset_items(Path("data/candidates"), r"^(cand_\d+)_cv$", unique_group=1)
    return max(1, cv_count), max(1, jobs_count)


def _show_timings(metrics: dict) -> None:
    timings = metrics.get("timings") or {}
    if not timings:
        return
    rows = [
        {"Stage": "Load", "Seconds": timings.get("load_seconds", 0)},
        {"Stage": "Filter", "Seconds": timings.get("filter_seconds", 0)},
        {"Stage": "Embed", "Seconds": timings.get("embed_seconds", 0)},
        {"Stage": "Match", "Seconds": timings.get("match_seconds", 0)},
        {"Stage": "Explain", "Seconds": timings.get("explain_seconds", 0)},
        {"Stage": "Total", "Seconds": timings.get("total_seconds", 0)},
    ]
    st.caption("Run timings")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _resolve_job_path(job_id: str) -> Path | None:
    for ext in (".txt", ".docx", ".pdf"):
        candidate = Path("data/jobs") / f"{job_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def _resolve_candidate_cv_path(candidate_id: str) -> Path | None:
    for ext in (".txt", ".docx", ".pdf"):
        candidate = Path("data/candidates") / f"{candidate_id}_cv{ext}"
        if candidate.exists():
            return candidate
    return None


def _read_doc_text(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return read_document_cached(path).strip()
    except Exception:
        return ""


def _snapshot_card(label: str, value: str) -> None:
    st.markdown(
        f"<div class='snapshot'><div class='snapshot-label'>{label}</div><div class='snapshot-value'>{value}</div></div>",
        unsafe_allow_html=True,
    )


def _build_export_df_with_explanations(df: pd.DataFrame) -> pd.DataFrame:
    export_df = df.copy()
    if "explanation" not in export_df.columns:
        export_df["explanation"] = ""
    explanation_text = export_df["explanation"].fillna("").astype(str).str.strip()
    missing_mask = explanation_text.eq("")
    if missing_mask.any():
        sim_series = pd.to_numeric(export_df.get("similarity_score", 0.0), errors="coerce").fillna(0.0)
        rank_series = export_df.get("candidate_rank", pd.Series(["-"] * len(export_df), index=export_df.index)).astype(str)
        job_title_series = export_df.get("job_title", pd.Series(["this role"] * len(export_df), index=export_df.index)).fillna("this role").astype(str)
        strength = pd.Series("partial alignment", index=export_df.index)
        strength = strength.mask(sim_series >= 0.55, "moderate alignment")
        strength = strength.mask(sim_series >= 0.75, "strong alignment")
        generated = (
            "Candidate shows "
            + strength
            + " for "
            + job_title_series
            + " (similarity="
            + sim_series.round(3).astype(str)
            + ", rank="
            + rank_series
            + ")."
        )
        export_df.loc[missing_mask, "explanation"] = generated.loc[missing_mask]
    return export_df


# Render ranking table, metrics, and detailed feedback.
def _show_results(df: pd.DataFrame, metrics: dict) -> None:
    st.markdown("## Ranked Results")
    display_df = df.copy()
    display_df = display_df.rename(columns={"rank": "candidate_rank", "similarity": "similarity_score"})
    if "candidate_id" in display_df.columns:
        display_df["candidate_id_raw"] = display_df["candidate_id"]
        display_df["candidate_id"] = display_df["candidate_id"].map(_candidate_label)

    filtered_df = display_df.copy()

    ordered_cols = [
        c
        for c in ["job_id", "job_title", "candidate_id", "candidate_rank", "similarity_score"]
        if c in display_df.columns
    ]
    ui_df = filtered_df[ordered_cols]
    if len(ui_df) > MAX_UI_ROWS:
        st.warning(
            f"Showing first {MAX_UI_ROWS:,} rows out of {len(ui_df):,} to keep the UI responsive."
        )
        ui_df = ui_df.head(MAX_UI_ROWS)
    overview_tab, rankings_tab, feedback_tab = st.tabs(["Overview", "Rankings", "Feedback"])
    with overview_tab:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Jobs", str(metrics.get("jobs_loaded", 0)))
        c2.metric("Candidates", str(metrics.get("candidates_ranked", 0)))
        c3.metric("Rows", str(len(filtered_df)))
        c4.metric("Runtime", f"{metrics.get('timings', {}).get('total_seconds', 0)}s")
        c5.metric("Top-K", str(int(filtered_df["candidate_rank"].max()) if "candidate_rank" in filtered_df.columns else "-"))
        cache_stats = metrics.get("cache_stats")
        if cache_stats:
            st.caption(
                f"Cache: hits {cache_stats.get('hits', 0)}, misses {cache_stats.get('misses', 0)}, "
                f"hit rate {cache_stats.get('hit_rate', 0.0) * 100:.1f}%"
            )
        if "explained_rows" in metrics:
            st.caption(f"Explained rows: {metrics.get('explained_rows', 0)}")
        _show_timings(metrics)

    with rankings_tab:
        st.dataframe(
            ui_df,
            use_container_width=True,
            column_config={
                "similarity_score": st.column_config.ProgressColumn(
                    "similarity_score",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.3f",
                ),
            },
        )
        export_df = _build_export_df_with_explanations(filtered_df)
        if "explanation" in filtered_df.columns and filtered_df["explanation"].fillna("").astype(str).str.strip().eq("").any():
            st.caption("Export note: generated explanations were added for rows missing model-generated text.")
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download rankings.csv", data=csv_bytes, file_name="rankings.csv", mime="text/csv")

    with feedback_tab:
        feedback_df = filtered_df.copy()
        if "similarity_score" in feedback_df.columns:
            feedback_df = feedback_df.sort_values("similarity_score", ascending=False)
        for row in feedback_df.head(20).to_dict(orient="records"):
            rank_value = row.get("candidate_rank", "-")
            title = f"{row.get('job_title', '')} ({row.get('job_id', '')}) -> {row.get('candidate_id', '')} [rank {rank_value}]"
            with st.expander(title):
                explanation = str(row.get("explanation", "")).strip()
                st.write(explanation if explanation else "No explanation available for this match.")
                raw_candidate_id = str(row.get("candidate_id_raw", "")).strip()
                job_id = str(row.get("job_id", "")).strip()
                cv_path = _resolve_candidate_cv_path(raw_candidate_id)
                job_path = _resolve_job_path(job_id)
                cv_text = _read_doc_text(cv_path)
                job_text = _read_doc_text(job_path)
                cv_tab, job_tab = st.tabs(["CV", "Job Description"])
                with cv_tab:
                    st.caption(str(cv_path) if cv_path else "CV file not found in data/candidates.")
                    if cv_text:
                        st.text_area(
                            "CV content",
                            value=cv_text,
                            height=260,
                            key=f"cv_{job_id}_{raw_candidate_id}",
                            disabled=True,
                            label_visibility="collapsed",
                        )
                    else:
                        st.write("No readable CV content found.")
                with job_tab:
                    st.caption(str(job_path) if job_path else "Job file not found in data/jobs.")
                    if job_text:
                        st.text_area(
                            "Job description content",
                            value=job_text,
                            height=260,
                            key=f"job_{job_id}_{raw_candidate_id}",
                            disabled=True,
                            label_visibility="collapsed",
                        )
                    else:
                        st.write("No readable job description content found.")


# Dataset-mode runner: loads prepared corpora and executes shared matcher pipeline.
def _run_dataset_mode(
    top_k: int,
    output_explanations: bool,
    explanation_top_n_jobs: int,
    use_cache: bool,
    max_jobs: int | None,
    max_candidates: int | None,
    sample_seed: int,
    write_outputs: bool,
) -> tuple[pd.DataFrame, dict]:
    with st.status("Running matcher", expanded=True) as status:
        if max_jobs or max_candidates:
            status.write(
                f"Loading batch: jobs={max_jobs or 'all'}, candidates={max_candidates or 'all'}..."
            )
        else:
            status.write("Loading full dataset...")
        df, metrics = run_matching(
            mode="dataset",
            candidate_doc_mode="cv_only",
            top_k=top_k,
            write_text_report=False,
            use_cache=use_cache,
            clear_cache=False,
            required_skills=None,
            location=None,
            salary_min=None,
            salary_max=None,
            allow_model_download=False,
            refresh_online_data=False,
            refresh_max_candidates=500,
            refresh_max_jobs=100,
            refresh_seed=42,
            refresh_force=False,
            output_explanations=output_explanations,
            explanation_top_n_jobs=explanation_top_n_jobs,
            write_outputs=write_outputs,
            use_faiss=False,
            max_jobs=max_jobs,
            max_candidates=max_candidates,
            sample_seed=sample_seed,
        )
        timings = metrics.get("timings", {})
        status.write(
            " | ".join(
                [
                    f"Load {timings.get('load_seconds', 0)}s",
                    f"Embed {timings.get('embed_seconds', 0)}s",
                    f"Match {timings.get('match_seconds', 0)}s",
                    f"Explain {timings.get('explain_seconds', 0)}s",
                ]
            )
        )
        status.update(
            label=f"Run completed in {timings.get('total_seconds', 0)}s",
            state="complete",
            expanded=False,
        )
    return df, metrics


# Streamlit entrypoint and sidebar controls.
def main() -> None:
    st.set_page_config(page_title="CN6000 Matcher", page_icon="📊", layout="wide")
    _inject_styles()

    st.markdown(
        """
        <div class='hero'>
          <h2 style='margin:0'>CN6000 Matcher</h2>
          <p style='margin:6px 0 2px 0;color:#334155;font-size:16px;'>Semantic candidate-to-job ranking with explainable feedback and rotating batches.</p>
          <div style='margin-top:8px'>
            <span class='chip'>CV-job semantic matching</span>
            <span class='chip'>LinkedIn jobs dataset</span>
            <span class='chip'>Explainable feedback</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset_max_candidates, dataset_max_jobs = _current_dataset_limits()
    if "subset_index" not in st.session_state:
        st.session_state["subset_index"] = 0
    if "batch_seed" not in st.session_state:
        st.session_state["batch_seed"] = random.SystemRandom().randint(1, 999999)

    s1, s2 = st.columns(2)
    with s1:
        _snapshot_card("Dataset Jobs", f"{dataset_max_jobs:,}")
    with s2:
        _snapshot_card("Dataset Candidates", f"{dataset_max_candidates:,}")

    st.markdown(
        "<div class='tips'><strong>Usage tip:</strong> Click <em>Next batch</em> between runs to quickly explore new job/candidate slices while preserving speed.</div>",
        unsafe_allow_html=True,
    )

    st.sidebar.header("Run settings")
    top_k = st.sidebar.slider("Top-K jobs per candidate", min_value=1, max_value=50, value=3)
    use_cache = USE_EMBEDDING_CACHE
    output_explanations = GENERATE_EXPLANATIONS
    subset_mode = USE_SUBSET_MODE
    next_batch_clicked = False
    if subset_mode:
        ratio = st.sidebar.selectbox(
            "Candidates per job ratio",
            options=[5, 3, 2],
            index=0,
            format_func=lambda x: f"{x}x",
        )
        subset_jobs = int(
            st.sidebar.number_input(
                "Jobs per run",
                min_value=1,
                max_value=dataset_max_jobs,
                value=min(DEFAULT_SUBSET_JOBS, dataset_max_jobs),
                step=1,
            )
        )
        min_candidates = min(dataset_max_candidates, subset_jobs * ratio)
        max_candidates_allowed = dataset_max_candidates
        default_candidates = min(dataset_max_candidates, subset_jobs * ratio)
        if subset_jobs * ratio > dataset_max_candidates:
            st.sidebar.caption(
                f"Available candidates ({dataset_max_candidates}) are below {ratio}x for {subset_jobs} jobs."
            )
        subset_candidates = int(
            st.sidebar.number_input(
                "Candidates per run",
                min_value=max(1, min_candidates),
                max_value=max_candidates_allowed,
                value=max(1, default_candidates),
                step=1,
            )
        )
        if st.sidebar.button("Next batch"):
            next_batch_clicked = True
            st.session_state["subset_index"] = int(st.session_state["subset_index"]) + 1
            st.session_state["batch_seed"] = random.SystemRandom().randint(1, 999999)
            st.session_state["batch_notice"] = "Next batch loaded. Running matcher for this batch..."
        active_seed = int(st.session_state["batch_seed"])
        st.sidebar.caption(f"Current batch index: {st.session_state['subset_index']}")
        st.sidebar.caption(f"Candidates/jobs ratio this run: {subset_candidates/max(subset_jobs,1):.2f}x")
        if "batch_notice" in st.session_state:
            st.sidebar.success(st.session_state.pop("batch_notice"))
        max_jobs = subset_jobs
        max_candidates = subset_candidates
        write_outputs = False
    else:
        max_jobs = None
        max_candidates = None
        active_seed = BASE_SUBSET_SEED
        write_outputs = True

    explanation_top_n_jobs = 0 if output_explanations else 0

    run_requested = st.button("Run matcher", type="primary") or next_batch_clicked
    if run_requested:
        try:
            df, metrics = _run_dataset_mode(
                top_k=top_k,
                output_explanations=output_explanations,
                explanation_top_n_jobs=explanation_top_n_jobs,
                use_cache=use_cache,
                max_jobs=max_jobs,
                max_candidates=max_candidates,
                sample_seed=active_seed,
                write_outputs=write_outputs,
            )
            st.session_state["last_df"] = df
            st.session_state["last_metrics"] = metrics
        except Exception as exc:
            st.error(f"Run failed: {exc}")

    if "last_df" in st.session_state and "last_metrics" in st.session_state:
        _show_results(st.session_state["last_df"], st.session_state["last_metrics"])


if __name__ == "__main__":
    main()
