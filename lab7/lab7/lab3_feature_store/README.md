# Lab 3: Feature Store co ban voi Feast

## Muc tieu

Xay dung feature store cho credit scoring:

- Tao feature definitions.
- Store offline va online features.
- Feature retrieval API.
- Reuse features cho training va serving.

## Cai dat

```powershell
cd "C:\Nam 4\CNM\lab7"
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r .\lab3_feature_store\requirements.txt
```

## Feast workflow

```powershell
cd .\lab3_feature_store\feature_repo
feast apply
python ..\scripts\materialize_features.py
```

## Chay API

```powershell
cd ..
uvicorn app:app --reload --port 8000
```

Test:

- `GET /health`
- `GET /features/1001`
