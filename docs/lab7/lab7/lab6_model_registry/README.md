# Lab 6: Model Versioning voi MLflow Model Registry

## Muc tieu

Version model bang MLflow Model Registry:

- Save model versions.
- Tag model.
- Rollback.
- Registry.

## Cau truc

```text
lab6_model_registry/
|-- artifacts/
|-- data/
|   `-- credit_default.csv
|-- mlruns/
|-- README.md
|-- requirements.txt
`-- src/
    |-- list_versions.py
    |-- promote_model.py
    |-- register_model.py
    `-- rollback_model.py
```

## Cai dat

```powershell
cd "C:\Nam 4\CNM\lab7"
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r .\lab6_model_registry\requirements.txt
```

## Dang ky model version moi

```powershell
python .\lab6_model_registry\src\register_model.py --model-name credit_default_model --variant logistic
python .\lab6_model_registry\src\register_model.py --model-name credit_default_model --variant tree
```

## Xem registry

```powershell
python .\lab6_model_registry\src\list_versions.py --model-name credit_default_model
```

## Promote version tot nhat

```powershell
python .\lab6_model_registry\src\promote_model.py --model-name credit_default_model --version 2 --alias champion --tag env=prod
```

## Rollback

```powershell
python .\lab6_model_registry\src\rollback_model.py --model-name credit_default_model --version 1 --alias champion
```

## Giao dien MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///C:/Nam%204/CNM/lab7/lab6_model_registry/mlflow.db --port 5002
```

Mo trinh duyet tai `http://127.0.0.1:5002`.

## Ghi chu

- Registry dung backend SQLite cuc bo.
- Moi version duoc log params, metrics va tag.
- Alias `champion` dung de chon model hien tai.
- Rollback chi can doi alias tro ve version truoc do.
