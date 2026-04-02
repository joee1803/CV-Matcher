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
5. Deploy.

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
1. local `data/jobs` and `data/candidates`
2. Hugging Face dataset snapshot
3. tracked `demo_data`
