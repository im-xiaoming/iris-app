# Lab 5: Hyperparameter Tuning Pipeline voi Optuna

## Muc tieu

Toi uu model bang Optuna voi day du cac thanh phan:

- Grid search va Bayesian optimization.
- Parallel runs.
- Logging trial va best run.
- Best model selection.

## Cau truc

```text
lab5_optuna/
|-- artifacts/
|-- data/
|   `-- customer_risk.csv
|-- README.md
|-- requirements.txt
`-- src/
    |-- compare_trials.py
    `-- tune_model.py
```

## Cong nghe

- Optuna
- MLflow
- scikit-learn

## Cai dat

```powershell
cd "C:\Nam 4\CNM\lab7"
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r .\lab5_optuna\requirements.txt
```

## Chay tuning

```powershell
python .\lab5_optuna\src\tune_model.py --search bayesian --trials 12 --jobs 2
python .\lab5_optuna\src\tune_model.py --search grid
```

## So sanh trial

```powershell
python .\lab5_optuna\src\compare_trials.py
```

## MLflow UI

```powershell
mlflow ui --backend-store-uri .\lab5_optuna\mlruns --port 5001
```

Mo trinh duyet tai `http://127.0.0.1:5001`.

## Ket qua duoc tao ra

- `artifacts/trials_<search>.csv`: toan bo trial.
- `artifacts/best_params_<search>.json`: tham so tot nhat.
- `artifacts/best_model_<search>/`: model tot nhat luu bang MLflow.
- Log params, metrics, tags va nested runs trong MLflow.
