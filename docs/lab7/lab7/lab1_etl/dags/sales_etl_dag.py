from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from extract import extract_from_api, extract_from_csv
from load import load_all
from transform import transform


def run_etl(**context):
    run_id = f"airflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    csv_data = extract_from_csv(run_id)
    api_data = extract_from_api(run_id)
    transformed = transform(csv_data, api_data, run_id)
    load_all(transformed, run_id)
    context["ti"].xcom_push(key="run_id", value=run_id)


default_args = {
    "owner": "data_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="sales_etl_pipeline",
    description="ETL pipeline for retail sales CSV + API to PostgreSQL",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["etl", "sales", "airflow"],
) as dag:
    etl_task = PythonOperator(
        task_id="run_sales_etl",
        python_callable=run_etl,
        provide_context=True,
    )

    etl_task
