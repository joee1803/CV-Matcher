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
4. Ensure repo includes required data folders/files:
- `data/jobs`
- `data/candidates`
5. Deploy.

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

- This app is designed for local dataset files (`data/jobs`, `data/candidates`).
- If hosting remotely, make sure those datasets are present in the deployment environment.
