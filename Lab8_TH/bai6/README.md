# Bai 6

## Chay nhanh

```powershell
cd D:\lab_CNM\Lab8_TH\bai6
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\data.py
python src\train.py
python src\deploy.py
python app.py
```

## MLflow

```powershell
docker compose up -d mlflow
```

- MLflow: `http://127.0.0.1:5005`
- root: `http://127.0.0.1:8002/`
- API: `http://127.0.0.1:8002/ping`
- report: `http://127.0.0.1:8002/report`

## Airflow

```powershell
python -m venv .venv_air
.venv_air\Scripts\activate
pip install -r requirements_airflow.txt
$env:AIRFLOW_HOME="$PWD\\.airflow"
$env:AIRFLOW__CORE__DAGS_FOLDER="$PWD\\dags"
airflow standalone
```
