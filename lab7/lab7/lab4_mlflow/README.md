# Lab 4: Experiment Tracking voi MLflow

## Muc tieu

Quan ly thi nghiem huan luyen mo hinh bang MLflow:

- Log parameters.
- Log metrics.
- Compare models.
- Visualize results.

## Cau truc

```text
lab4_mlflow/
|-- artifacts/
|-- data/
|   `-- customer_churn.csv
|-- mlruns/
|-- README.md
|-- requirements.txt
`-- src/
    |-- compare_runs.py
    `-- train_experiments.py
```

## Cai dat

```powershell
cd "C:\Nam 4\CNM\lab7"
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r .\lab4_mlflow\requirements.txt
```

## Chay thi nghiem

```powershell
python .\lab4_mlflow\src\train_experiments.py
```

Script se:

- Train nhieu mo hinh.
- Log params, metrics, model va artifact vao MLflow.
- Tao file tong hop ket qua de so sanh run.

## So sanh ket qua

```powershell
python .\lab4_mlflow\src\compare_runs.py
```

## Mo giao dien MLflow

```powershell
mlflow ui --backend-store-uri .\lab4_mlflow\mlruns --port 5000
```

Mo trinh duyet tai `http://127.0.0.1:5000`.

## Cac thong tin duoc log

- Parameters: ten model, random_state, test_size, max_depth, C.
- Metrics: accuracy, precision, recall, f1, roc_auc.
- Artifacts: `run_summary.csv`.
- Model: moi run deu log mo hinh da train.
