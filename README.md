
# FraudOps - Credit Card Fraud Detection MLOps Demo

FraudOps is a portfolio-grade MLOps project for credit card fraud detection. It combines a FastAPI prediction service, a model registry/admin layer, versioned model metadata, Dockerized serving, and a lightweight browser UI for transaction scoring.

The project is built to demonstrate practical data science and MLOps skills:

- fraud classification on the public Kaggle credit card fraud dataset
- model versioning and production approval metadata
- FastAPI serving with `/predict`, `/health`, `/model-info`, and OpenAPI docs
- Docker deployment for a live web demo
- DVC/MLflow-oriented retraining and experiment tracking workflow

## Live Demo

Production serving demo: https://fraudops-mlops-demo.onrender.com

API health check: https://fraudops-mlops-demo.onrender.com/health

Admin/model registry demo: https://fraudops-admin-demo.onrender.com

The admin demo runs in portfolio mode on free hosting: parameter changes create traceable candidate versions and registry metadata without launching the heavier DVC/MLflow retraining job.

## Live Demo Deployment

The recommended free deployment path is the **serving API** using Render Free Web Services and `Dockerfile.serving`.

See [DEPLOYMENT.md](DEPLOYMENT.md) for the full setup.

For public portfolio deployment, large pickle artifacts are intentionally excluded from Git. The serving app can fall back to the trained logistic regression coefficients stored in `models/production/model_summary.txt`, keeping the repository lightweight while still exposing a real model-derived scoring function.

Structure
--------------------

    .
    ├── .dvc <- config files of dvc
    ├── .github/workflows
        ├── cml.yaml <- docker continer action, that pull data from storage, reproduce the pipeline and share the results 
    ├── src
        ├── Resampling.py <- functions that use resample technics (under/over-sampling & SMOTE)
        ├── Stepwise_model_selection.py <- selection of the best accurate model (logit or regression) 
        ├── data_understanding.py <- scripts that generate description of the raw data
        ├── prepare.py <- auxiliary functions and classes
        ├── train.py <- training the model on the output prepare
        └── test.py <- testing the model on the new sample and the whole original data-set (using pickle files)
    ├── templates
        └── index.html <- html script for the ML Product 
    ├── static
        └── css
           └── style.css   
    ├── .dvcignore <- contains name of untracked files by dvc existing remote   
    ├── .gitignore  
    ├── README.md <- this file :)
    ├── creditcard.csv.dvc <- tracked dataset
    ├── dvc.lock  <-  restricts access to stages outputs
    ├── dvc.yaml <- contains stages that create the pipeline
    ├── params.yaml <- contains all the parameters used in the src/ scripts see more on description below
    ├── report.md
    └── requirements.txt
Pipeline        
-------------------- 

                              | creditcard.csv.dvc |
                             *************************  *******
                       ******             *                  *********
                  *****                   *                           *******
               ***                        *                                  *********
                                                                                     ****
    | understand data |                | resample data |                                 *
                                                                                         *
                                          *                                              *
                                          *                                              *
                                          *                                              *
                                                                                         *
                                  | splitting data |                                     *
                                                                                         *
                                 ***            ***                                      *
                               **                  ***                                   *
                             **                       **                                 *
                                                        **                             ***
                    | train model |                         *                        *****
                                  *****                  *                   *****
                                       ******            *             ******
                                             *****       *        *****
                                                  ***    *     ***
                                                         *
                                                    | test model |   
                                                              
##### run this cmd (after doing Preparation) to see the pipeline 
```bash
dvc dag
```
-------------------- 
# Preparation

### 1. Clone this repository
```bash
git clone https://github.com/MedEleliem/Fraud_Credit_Card
```
### 2. Get data

Download creditcard.csv

```bash
dvc pull creditcard.csv
``` 
#### You can modify the params.yaml only, it contains all the parameters used in the model production, or you can modify the scripts file in src/

### 3. Restart the pipeline
```bash
dvc repro
```
#### after doing this, push it to a new branche, differences and new plots will be displayed in the github-action bot in the new pull-request

### Also you can simply edit on github, create a pull resquest, the workflow is already automated ;) 
### e.g
![image](https://user-images.githubusercontent.com/64113527/118899364-7a6f7800-b906-11eb-96a4-097917ca385c.png)

--------------------
# Two-Server MLOps Platform

The project is now split into two separate FastAPI services:

```text
Admin / MLOps server
  - parameter drafts
  - DVC retraining
  - MLflow tracking
  - model comparison
  - approval for serving

Client / Serving server
  - manual transaction input UI
  - /predict API
  - loads only an approved production model
  - no DVC, MLflow, or admin controls
```

The registry intentionally starts empty:

```json
{
  "active_version": null,
  "versions": []
}
```

The client server returns `waiting_for_model` until an MLOps admin approves a candidate and publishes the serving image.

## Admin server

The new serving layer starts with a model registry UI and metadata API. It is separated from the historical training dependencies so the real-time service can evolve cleanly.

```bash
python3 -m venv .venv-api
source .venv-api/bin/activate
pip install -r requirements-api.txt
pip install -r requirements-mlflow.txt
uvicorn app.main:app --reload
```

Available routes:

```text
GET /                 model registry UI
GET /health           service health check
GET /api/models       registered model versions
GET /api/models/active active model metadata
GET /api/models/{version} model version metadata
GET /api/models/{candidate}/diff/{baseline} model comparison
GET /api/pipeline/results final DVC metrics, reports, and plot readiness
GET /api/admin/state admin drafts and retraining jobs
POST /api/admin/drafts create a parameter draft
POST /api/admin/drafts/{draft}/submit create a candidate snapshot and retraining job
POST /api/admin/models/{version}/approve approve a model for client serving
GET /docs             FastAPI OpenAPI docs
```

## Client serving server

```bash
pip install -r requirements-serving.txt
uvicorn serving_app.main:app --host 127.0.0.1 --port 8080
```

Available routes:

```text
GET /             manual transaction prediction UI
GET /health       serving status and model readiness
GET /model-info   approved model metadata
POST /predict     final fraud prediction
```

The client payload accepts `Time`, `Amount`, and `V1` through `V28`. If no model has been approved into `models/production`, `/predict` returns `503`.

## Model artifact versioning

Every model parameter change should produce a new immutable model version instead of replacing the current pickle files. Admins can now use the platform UI to prepare parameters, save a draft, validate it, and submit a retraining request. The UI creates a candidate snapshot immediately; CI/DVC should then generate the real artifacts and metrics for that version.

The local/CI flow remains:

```bash
dvc repro
python scripts/register_model.py --version fraud-logit-my-change --copy-artifacts
python scripts/log_mlflow_run.py --run-name fraud-logit-my-change
python scripts/compare_models.py fraud-logit-my-change fraud-logit-2021-05-18
```

Each version is stored under:

```text
models/versions/{version}/
  manifest.json
  params.yaml
  metrics.json
  logit.pkl
  finalvar.pkl
  model_summary.txt
```

The comparison checks parameter diffs, metric deltas, feature changes, artifact paths, and a simple gate for fraud-class regressions. GitHub Actions can run the same scripts after `dvc repro` and upload `model-diff.json` for pull-request review.

## MLflow tracking

MLflow stores the final training run after the DVC pipeline has produced real outputs:

```bash
pip install -r requirements-mlflow.txt
dvc repro
python scripts/log_mlflow_run.py --run-name fraud-logit-my-change
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The MLflow run logs:

```text
params.yaml values
accuracy and fraud-class metrics
classification reports
model summary
resampling plots
confusion matrices
model and selected-feature pickles
```

The platform UI shows a live parameter preview before training, then the final DVC/MLflow metrics and plots after the retraining run completes.

## Docker continuous delivery

The platform uses separate images for training/admin and serving:

```bash
docker compose -f docker-compose.admin.yml up fraud-admin
docker compose -f docker-compose.admin.yml --profile training run fraud-training
docker compose -f docker-compose.serving.yml up fraud-serving
```

After a candidate is approved:

```bash
python scripts/build_serving_image.py --version fraud-logit-YYYY-MM-DD-xxxxx
```

In production, CI should push `fraud-serving:{model_version}` to a Docker registry. The client serving server then pulls that immutable image and restarts, giving a clean continuous delivery path from approved model to final prediction API.

## Kaggle dataset

The dataset used by this project is Kaggle `mlg-ulb/creditcardfraud`, the common European cardholders fraud dataset with 284,807 transactions and 492 frauds. To download it for real local runs:

```bash
pip install -r requirements-data.txt
python scripts/download_kaggle_data.py
```

Kaggle may require API credentials in `~/.kaggle/kaggle.json`.

## Admin workflow

```text
Open the platform
-> edit model parameters
-> save a draft
-> validate and submit retraining
-> candidate snapshot is created
-> CI/DVC produces artifacts
-> diff and readiness gate are reviewed
-> candidate can be promoted later
```

--------------------
# Description

#### when you are a Data-Scientist, and you are building a ML model, for example scores and plot are necessarily for showing the power of your model, supposing that you are working with a team of data-scientists and they want to get a branch of your original project in order to test the model locally and correct some errors or doing some regularizations or any other modifications that will improve your model optimality, they want to know the result of their commitments. So, you need to regenerate txt files that contains F1 scores for example or Students statistics TEST of the coefficients in case of linear regression or maybe a printed image file saved as jpg or png for a boxplot or residuals plot or scatter-plot in the python file that train your data-set, this files needs to be updated in each commitment, you need to create a YAML file coded in GO language in Docker software in every push or pull.
#### What if Dataset is too big:
#### When we have a large CSV file or a data that contains images or sound files, evaluating the model on github will be impossible because we cannot upload it on github, a new topic is introduced is DVC, DVC is built to make ML models shareable and reproducible. It is designed to handle large files, data sets, machine learning models, and metrics as well as code. So DVC can provide us to upload files on drives platforms (Google Drive for example) and linked it to our github repositories, so when one of your team want to pull the project, he will be able to download the shared Dataset also the other python files then evaluate the model locally on his machine

--------------------
# Want to do some changes ?

### params.yaml
#### get into this file

### to choose resampling technics
#### splitting_data : 
    split-param : 1 or 2 or 3 , Undersampling or Oversampling or SMOTE
    param : [0;1]
    
### To change the train/test split percentage : 
#### splitting_data :
    split-param : 0.2

#### To change the model stepwise selection Paramaters :
##### train_model :
       model_type : "logistic" or "regression"
       elimination_criteria : "aic" or "bic"
       varchar_process : "dummy_dropfirst" 
       p-value :  0.05, I dont know if there such a elimination criteria better than this <3 
