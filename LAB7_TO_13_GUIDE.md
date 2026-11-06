# Remaining Labs — execution guide

| Lab | Topic | Main files | Command / validation |
|---|---|---|---|
| 7 | Pipeline engineering | `src/extended/pipeline.py` | `python -m src.extended.pipeline` |
| 8 | CI automation | `.github/workflows/wine-quality-ci.yml` | Push to GitHub / Actions |
| 9 | Failure handling + quality gate | `src/extended/failure_test.py`, `src/extended/pipeline.py` | Run failure test; inspect `reports/quality_gate_report.json` |
| 10 | Model API | `api/app.py` | `uvicorn api.app:app --reload` |
| 11 | Docker + Kubernetes | `Dockerfile`, `deployment/kubernetes.yaml` | Docker build/run; `kubectl apply -f deployment/kubernetes.yaml` |
| 12 | Monitoring / drift | `src/extended/monitor_drift.py` | `python -m src.extended.monitor_drift` |
| 13 | End-to-end integration | `pipeline.py` + reports + registry | Run pipeline, then API and monitoring |

## Local validation

From the project root, activate the project virtual environment and run:

```bash
python setup.py
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.extended.failure_test
python -m src.extended.pipeline
python -m src.extended.inference
python -m src.extended.monitor_drift
```

Labs 3–6 can be run with `python -m pipelines.run_lab3_baseline` through `python -m pipelines.run_lab6_registry`. The pipeline validates the dataset schema and values, saves model and preprocessing artifacts, writes evaluation and quality-gate reports, registers the model, and promotes only a passing version. The production report is based on the MLflow registry state rather than a fixed status.

The API accepts feature names as they appear in the dataset or the corresponding underscore-style Python field names. It returns HTTP 503 from `/health` and `/predict` until both trained artifacts exist.

Do not report Docker/Kubernetes deployment or remote CI as completed until you have actually run and verified those environments.
