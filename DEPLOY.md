# Deployment Guide

## 1) Local deployment (recommended baseline)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 2) Streamlit Community Cloud

1. Push project to GitHub.
2. In Streamlit Community Cloud, create app from repo.
3. Set:
- Main file path: `app.py`
- Python version: `3.11`
4. Configure one dataset source:
- local `data/jobs` and `data/candidates`
- or a Hugging Face dataset snapshot
- or the tracked fallback `demo_data`
- or provide the dataset path in app settings if the full dataset is stored outside the repo:
- `DATA_ROOT=/absolute/path/to/dataset_root`
  - expected structure under that folder:
    - `jobs/`
    - `candidates/`
5. Or provide explicit folders instead:
- `JOBS_DATA_DIR=/absolute/path/to/jobs`
- `CANDIDATES_DATA_DIR=/absolute/path/to/candidates`
6. Deploy.

### Hugging Face dataset settings

Add these environment variables in Streamlit Community Cloud if you want a larger deployment dataset from Hugging Face:

- `HF_DATASET_REPO_ID`: for example `your-name/cv-matcher-data`
- `HF_DATASET_REPO_TYPE`: usually `dataset`
- `HF_DATASET_REVISION`: optional branch, tag, or commit
- `HF_DATASET_SUBDIR`: optional subfolder that contains `jobs/` and `candidates/`
- `HF_TOKEN`: optional, only needed for private repos

Expected Hugging Face layout:

```text
jobs/
  jd_0001.txt
  ...
candidates/
  cand_0001_cv.txt
  ...
```

## 3) Render (alternative)

1. Create new Web Service from repo.
2. Build command:

```bash
pip install -r requirements.txt
```

3. Start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## 4) Deployment checks

- App opens successfully.
- `Run matcher` completes on current dataset.
- `Next batch` loads and triggers a new run.
- `Download rankings.csv` works and includes `explanation` for all rows.

## 5) Notes

- Dataset resolution order is:
  1. `JOBS_DATA_DIR` + `CANDIDATES_DATA_DIR`
  2. `DATA_ROOT/jobs` + `DATA_ROOT/candidates`
  3. repo-local `data/jobs` + `data/candidates`
  4. Hugging Face dataset snapshot
  5. repo-local `demo_data/jobs` + `demo_data/candidates`
- If hosting remotely, the full dataset must exist in the deployment environment or be mounted from external storage.
