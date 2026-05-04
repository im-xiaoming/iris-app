# Bai 2

## Chay bai

```powershell
cd D:\lab_CNM\Lab8_TH\bai2
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m dvc init --no-scm -f
python -m dvc repro
```

## File chinh

- model: `out/model.pkl`
- metric: `out/metrics.json`
- env: `out/env.txt`
