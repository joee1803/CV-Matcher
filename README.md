# CN6000 Matcher

Semantic candidate-to-job matching with a Streamlit interface and local dataset files.

## Current runtime model

- Frontend: Streamlit (`app.py`)
- Backend pipeline: `src/run.py`
- Data source: local files only
  - jobs: `data/jobs/`
  - candidates: `data/candidates/`
  - deployment override: `DATA_ROOT`, or `JOBS_DATA_DIR` + `CANDIDATES_DATA_DIR`
- Batch sampling: random batch per run (with `Next batch`)

## Features

- Embedding-based ranking (`SentenceTransformer` + cosine similarity)
- Explainable match text per row
- CSV export with explanations for every exported row
- View CV and job description in the UI for top matches
- Cache-backed embeddings for faster reruns

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run app

```bash
streamlit run app.py
```

## Deployment dataset overrides

You can point the app at an external dataset location without committing `data/` into the repo.

Option 1:

```bash
DATA_ROOT=/absolute/path/to/dataset_root
```

Expected structure:

```text
dataset_root/
  jobs/
  candidates/
```

Option 2:

```bash
JOBS_DATA_DIR=/absolute/path/to/jobs
CANDIDATES_DATA_DIR=/absolute/path/to/candidates
```

## Data format

- Jobs folder: `data/jobs`
  - files: `jd_XXXX.txt|docx|pdf`
- Candidates folder: `data/candidates`
  - CV file required: `cand_XXXX_cv.txt|docx|pdf`
  - cover files optional: `cand_XXXX_cover.txt|docx|pdf`

## Main outputs

- `outputs/rankings.csv`
- `outputs/report.docx` (only when output writing is enabled by run mode)
- `outputs/.cache/*.npy` (embedding cache)
