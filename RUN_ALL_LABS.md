# Wine Quality MLOps — Labs 1 to 13

This is a separate, expanded Wine Quality project. The original Labs 1–6 files from the main `mlops cla` project are retained. The additional Lab 7–13 implementation is added under `src/extended`, `api`, `deployment`, and `.github/workflows`. The second supplied archive was a Telco churn project, so its later MLOps concepts have been adapted to the Wine Quality regression problem rather than copied with churn-specific columns/metrics.

## Lab mapping
- Labs 1–6: preprocessing, baseline evaluation, MLflow tracking, validation, registry, and lifecycle scripts under `src/` and `pipelines/`.
- Lab 7: end-to-end pipeline — load, validate, preprocess, train, evaluate, MLflow tracking and registry.
- Lab 8: GitHub Actions CI workflow.
- Lab 9: failure checks and quality-gate blocking.
- Lab 10: FastAPI serving (`/`, `/health`, `/predict`).
- Lab 11: Dockerfile, container execution, Kubernetes Deployment and Service.
- Lab 12: statistical drift monitoring using the Kolmogorov–Smirnov test.
- Lab 13: integrated pipeline, model artifacts, reports, registry and monitoring.

## Run on macOS (VS Code terminal)
```bash
cd wine-quality-complete-mlops-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python setup.py
python -m pytest -q
python -m src.extended.failure_test
python -m src.extended.pipeline
```

Run Labs 3–6 with `python -m pipelines.run_lab3_baseline` through `python -m pipelines.run_lab6_registry`. Run commands from the project root.

The dataset is expected at `data/raw/winequality-red.csv`. The integrated pipeline validates the complete feature schema and numeric values, writes artifacts and reports, and blocks registration and promotion if the R² quality gate fails. The production report is generated from actual MLflow registry state.

## View MLflow
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```
Open `http://127.0.0.1:5000`.

## Run inference
```bash
python -m src.extended.inference
```

## Run drift monitoring
```bash
python -m src.extended.monitor_drift
```
The drift script compares two reproducible partitions of the included dataset. It is a demonstration of statistical drift detection, not evidence of drift in a live incoming production batch.

## Run API (after pipeline creates models)
```bash
uvicorn api.app:app --reload
```
Open `http://127.0.0.1:8000/docs`.

## Docker
```bash
python -m src.extended.pipeline
docker build -t wine-quality-api:latest .
docker run --rm -p 8000:8000 wine-quality-api:latest
```

## Kubernetes
Build/load the image into your local cluster, then:
```bash
kubectl apply -f deployment/kubernetes.yaml
kubectl get deployments,pods,services
kubectl port-forward service/wine-quality-service 8000:80
```

## Important notes
- Run the Lab 7–13 pipeline before starting the API; it creates the isolated API artifacts `models/production_preprocessor.pkl` and `models/production_model.pkl`.
- MLflow registry behavior depends on the installed MLflow version and local tracking backend. Verify the registered model in the MLflow UI.
- The GitHub workflow is a starter CI workflow. Configure remote MLflow tracking separately if registry persistence across CI runs is required.
- Docker and Kubernetes steps require Docker Desktop and a working Kubernetes cluster (for example, Docker Desktop Kubernetes or Minikube).
- The quality gate uses an R² threshold of 0.30 as a configurable teaching/demo threshold. Review it against your instructor's rubric before submission.
- Docker and Kubernetes deployment and remote CI execution must be verified in those environments; local checks do not claim those deployments succeeded.
