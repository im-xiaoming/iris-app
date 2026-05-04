from datetime import datetime
import subprocess

from airflow import DAG
from airflow.operators.python import PythonOperator


dag = DAG(
    "fraud_mlops",
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
)


def run(cmd):
    subprocess.run(cmd, check=True)


t1 = PythonOperator(
    task_id="data",
    python_callable=run,
    op_args=[["python", "src/data.py"]],
    dag=dag,
)

t2 = PythonOperator(
    task_id="train",
    python_callable=run,
    op_args=[["python", "src/train.py"]],
    dag=dag,
)

t3 = PythonOperator(
    task_id="deploy",
    python_callable=run,
    op_args=[["python", "src/deploy.py"]],
    dag=dag,
)

t4 = PythonOperator(
    task_id="monitor",
    python_callable=run,
    op_args=[["python", "src/mon.py"]],
    dag=dag,
)

t1 >> t2 >> t3 >> t4
