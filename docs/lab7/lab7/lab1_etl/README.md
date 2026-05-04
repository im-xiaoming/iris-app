# Lab 1: ETL Pipeline cho du lieu ban hang

## Muc tieu

Xay dung pipeline ETL:

- Extract du lieu tu CSV va API.
- Clean missing values va outliers.
- Transform thanh schema chuan.
- Load vao Data Warehouse PostgreSQL.
- Logging toan bo pipeline.

## Chuan bi moi truong

Airflow `2.8.1` can Python `3.10` hoac `3.11`.

```powershell
cd "C:\Nam 4\CNM\lab7"
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r .\lab1_etl\requirements.txt
```

## PostgreSQL

Chay `psql` sau khi da cai PostgreSQL:

```powershell
& "C:\Program Files\PostgreSQL\16\bin\psql.exe" -U postgres
```

Sau khi vao `psql`, chay:

```sql
CREATE DATABASE sales_dw;
CREATE USER etl_user WITH PASSWORD 'etl_password';
GRANT ALL PRIVILEGES ON DATABASE sales_dw TO etl_user;
\q
```

Tao bang:

```powershell
& "C:\Program Files\PostgreSQL\16\bin\psql.exe" -U etl_user -d sales_dw -f .\lab1_etl\sql\create_tables.sql
```

## Chay script ETL

```powershell
python .\lab1_etl\scripts\run_pipeline.py --skip-db
```

Bo `--skip-db` neu ban da cau hinh PostgreSQL xong.

## Chay Airflow

```powershell
$env:AIRFLOW_HOME="$PWD\airflow_home"
airflow db init
airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin
airflow webserver --port 8080
airflow scheduler
```
