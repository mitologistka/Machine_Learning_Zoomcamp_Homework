# House Rent Prediction — Midterm Project

## Project overview
Predict monthly house rent from tabular inputs (area type, city, bhk, size, furnishing status, tenant preference, bathroom, point of contact, etc.). The solution includes EDA, feature engineering, several model experiments, a training script that exports a serialized model, a small Flask prediction API, tests and a Dockerfile for containerized serving.

## Problem statement (detailed)
Estimate the monthly rent for rental listings to help renters and listing platforms quickly gauge fair rental prices. Input is a single listing described by categorical and numeric features; output is a continuous numeric value (monthly rent). The model is intended for:
- Quick rent estimations in a UI or an API.
- Integration into dashboards or recommendation systems.
- Supporting human agents to price listings consistently.

Success criteria:
- Reasonable predictive accuracy on holdout data (lower RMSE / MAE, higher R²).
- Fast, deterministic inference in a lightweight API.
- Repeatable training and deployment.

## Dataset
- Source: public Kaggle house rent dataset (referenced in notebook).
- Typical features: city, area_type, size (sqft), bhk, bathroom, furnishing_status, tenant_preferred, point_of_contact, and price (target).
- Data handling: missing value checks, type conversions, and basic outlier filtering performed in notebook and training script.

## Exploratory Data Analysis (summary)
- Missing values: identified per-column and imputed or dropped depending on frequency.
- Ranges & distributions: numeric features (size, price) inspected; price shows right skew — log-transform used in some experiments.
- Target analysis: correlation of price with size, bhk, and locality; strong variance across cities.
- Categorical analysis: frequent categories inspected and rare categories grouped as "other".
- Feature importance: tree-based models used to assess top predictors (size, city, bhk, furnishing_status).

(Full EDA and visualizations in Project.ipynb.)

## Feature engineering
- Convert size and bhk to numeric where needed.
- One-hot encoding for small-cardinality categorical variables; target/impact encoding or grouping for high-cardinality locality features.
- Log transform for target when training regressors to stabilize variance (inverse transform at prediction).

## Models and training
- Models tried: linear models (LinearRegression, Ridge), tree-based (RandomForest), gradient boosting (XGBoost / LightGBM when available).
- Experiments: basic baseline linear model, multiple tree-based models, and ensembles where applicable.
- Hyperparameter tuning: grid / randomized search on key parameters for tree models (depth, n_estimators, learning rate where applicable).
- Final model: selected based on validation RMSE/MAE and generalization on holdout.

Training code:
- train.py encapsulates preprocessing, training, evaluation, and model serialization (model.bin).
- Notebook contains the exploratory workflows and ad-hoc experiments.

## Evaluation
- Metrics used: RMSE, MAE, R². Cross-validation and holdout validation used for robust estimates.
- Use logged or raw metrics depending on whether the target was transformed.

## Reproducibility
- Dependencies declared in Pipfile. Install with pipenv or create a virtualenv and install dependencies from the Pipfile.
- train.py expects the dataset to be present or to be downloaded as described in Project.ipynb.
- Running train.py produces model.bin used by the API.

## Serving & API
- predict.py: Flask app that loads the serialized model and exposes a /predict POST endpoint (JSON body -> predicted rent).
- test_predict.py: lightweight test for the prediction endpoint.

Example request:
curl -X POST http://0.0.0.0:9696/predict -H "Content-Type: application/json" -d '{"bhk":1,"size":600,"area_type":"carpet_area","city":"bangalore","furnishing_status":"unfurnished","tenant_preferred":"bachelors","bathroom":2,"point_of_contact":"contact_owner"}'

## Docker
- Dockerfile provided to build a container with the API and model file.
- Build & run:
  docker build -t house-rent-predict .
  docker run -p 9696:9696 house-rent-predict

## How to run (quick)
1. Setup environment:
   - pipenv install --dev
   or
   - python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt (generate from Pipfile if needed)
2. Train:
   - python train.py
3. Serve:
   - python predict.py
4. Test:
   - pytest test_predict.py

## Results & limitations
- Models perform well on common locations but degrade on rare or unseen localities.
- Outliers and noisy listings (incorrect size or price) can affect performance; robust cleaning recommended.
- Model should be retrained periodically with fresh data to capture market shifts.

## Next steps
- Add automated Kaggle data download and data versioning.
- Add CI for tests and Docker build.
- Improve locality encoding (geospatial features) and include time-series trends.
- Add a small sample dataset to the repo for zero-friction reproducibility.

## Files of interest
- Project.ipynb — full EDA and experimentation.
- train.py — production training script.
- predict.py — Flask API for inference.
- test_predict.py — tests.
- Dockerfile, Pipfile — container and dependency management.
