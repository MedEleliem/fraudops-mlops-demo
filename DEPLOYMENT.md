# FraudOps Deployment Guide

This repository is designed to showcase an MLOps fraud detection workflow:

- an admin/model registry application under `app/`
- a lightweight serving API and prediction UI under `serving_app/`
- immutable model metadata under `models/versions/`
- a production model package under `models/production/`

## Recommended free demo hosting

Use Render Free Web Service for the live demo. It supports Docker web services on a free instance, which fits this project better than platforms that only support static sites or Streamlit apps.

The live demo should run the serving API only:

```bash
uvicorn serving_app.main:app --host 0.0.0.0 --port 8080
```

The admin/DVC/MLflow side is intentionally kept as source code and documentation because retraining jobs and MLflow tracking are heavier than a small free web service should run.

## Why the demo works without pickle artifacts

Large model pickle files are excluded from Git by `.gitignore`. For portfolio deployment, the serving API falls back to the trained logistic regression coefficients stored in:

```text
models/production/model_summary.txt
```

This keeps the public repository light while still serving a real model-derived scoring function.

## Deploy on Render

1. Push this repository to GitHub.
2. Create a Render account.
3. Choose **New > Web Service**.
4. Connect the GitHub repository.
5. Select the free plan.
6. Use Docker deployment. Render will detect `render.yaml`, or you can select:

```text
Dockerfile: Dockerfile.serving
Health check path: /health
```

7. Deploy and open the generated `.onrender.com` URL.

Useful endpoints:

```text
GET  /          prediction UI
GET  /health    service status
GET  /model-info deployed model metadata
POST /predict   fraud scoring API
GET  /docs      OpenAPI docs
```

## Local smoke test

```bash
pip install -r requirements-serving.txt
uvicorn serving_app.main:app --host 127.0.0.1 --port 8080
```

Then open:

```text
http://127.0.0.1:8080
```

## Portfolio positioning

Suggested GitHub description:

```text
FraudOps: FastAPI MLOps demo for credit card fraud detection with model registry metadata, versioned artifacts, Docker serving, and live transaction scoring UI.
```

Suggested Upwork portfolio line:

```text
Built a deployable fraud detection MLOps platform with FastAPI, Docker, model versioning, registry metadata, and a live scoring interface.
```
