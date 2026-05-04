# Lab 6 - End-to-End MLOps System

Lab nay tong hop mot he thong MLOps end-to-end, ket noi day du cac thanh phan tu du lieu den deployment va monitoring.

## Muc tieu

1. Data pipeline
2. Training pipeline
3. Deployment
4. Monitoring
5. CI/CD

## Cau truc thu muc

```text
lab6/
|-- .github/
|   `-- workflows/
|       `-- mlops-ci-cd.yml
|-- artifacts/
|-- data/
|   |-- processed/
|   `-- raw/
|-- deployment/
|   `-- current/
|-- monitoring/
|   |-- logs/
|   |-- reports/
|   `-- generate_monitoring_report.py
|-- pipelines/
|   |-- data_pipeline.py
|   |-- deployment_pipeline.py
|   `-- training_pipeline.py
|-- service/
|   `-- main.py
|-- tests/
|   `-- test_training_metrics.py
|-- docker-compose.yml
|-- dvc.yaml
|-- Jenkinsfile
|-- params.yaml
|-- README.md
`-- requirements.txt
```

## Yeu cau va cach dap ung

### 1. Data pipeline

`pipelines/data_pipeline.py` thuc hien:

- Nap dataset tu `scikit-learn`
- Chuyen target thanh nhan de doc
- Luu raw data vao `data/raw/`
- Chia train/test va luu vao `data/processed/`
- Co dinh seed de ket qua lap lai

Pipeline nay duoc mo ta trong `dvc.yaml` de co the chay lai theo stage.

### 2. Training pipeline

`pipelines/training_pipeline.py`:

- Doc train/test data da xu ly
- Train model `RandomForestClassifier`
- Danh gia accuracy va F1
- Luu model vao `artifacts/model.joblib`
- Luu metrics vao `artifacts/training_metrics.json`

### 3. Deployment

`pipelines/deployment_pipeline.py`:

- Lay model tu `artifacts/`
- Copy sang `deployment/current/model.joblib`
- Day la vi tri model duoc service su dung

FastAPI service trong `service/main.py` doc model deployment nay de phuc vu prediction.

### 4. Monitoring

Co 2 lop monitoring:

- Runtime monitoring trong `service/main.py`
- Offline monitoring trong `monitoring/generate_monitoring_report.py`

Runtime monitoring bao gom:

- Log moi request prediction vao JSONL
- Do latency request
- Expose `/metrics` theo Prometheus

Offline monitoring bao gom:

- Tong hop so request
- Tinh latency trung binh
- Thong ke phan bo prediction
- Tao placeholder drift report

### 5. CI/CD

He thong co 2 cau hinh CI/CD:

- GitHub Actions: `.github/workflows/mlops-ci-cd.yml`
- Jenkins: `Jenkinsfile`

Cac buoc tu dong gom:

1. Cai dependency
2. Chay data pipeline
3. Chay training pipeline
4. Chay test
5. Deploy model
6. Tao monitoring report

## DVC Pipeline

File `dvc.yaml` dinh nghia 3 stages:

1. `data_pipeline`
2. `training_pipeline`
3. `deployment_pipeline`

Co the re-run:

```bash
cd lab6
dvc repro
```

## FastAPI Service

Chay service:

```bash
cd lab6
uvicorn service.main:app --reload
```

Endpoints:

- `GET /health`
- `POST /predict`
- `GET /metrics`

## Cach chay tung buoc

### Local

```bash
cd lab6
pip install -r requirements.txt
python pipelines/data_pipeline.py
python pipelines/training_pipeline.py
python pipelines/deployment_pipeline.py
python monitoring/generate_monitoring_report.py
uvicorn service.main:app --reload
```

### Docker Compose

```bash
cd lab6
docker compose up
```

### Test

```bash
cd lab6
pytest tests
```

## Dau ra sau khi pipeline chay

- `data/raw/breast_cancer.csv`
- `data/processed/train.csv`
- `data/processed/test.csv`
- `artifacts/model.joblib`
- `artifacts/training_metrics.json`
- `deployment/current/model.joblib`
- `monitoring/logs/prediction_requests.jsonl`
- `monitoring/reports/service_metrics.json`
- `monitoring/reports/drift_report.json`

## Ghi chu

- Lab nay mo phong mot he thong MLOps tong hop de hoc tap va demo
- Co the mo rong them model registry, feature store, canary deployment, alerting, drift detection nang cao, va orchestration bang Airflow/Kubeflow
- Cau truc nay phu hop de trinh bay mot pipeline hoan chinh tu dau den cuoi
