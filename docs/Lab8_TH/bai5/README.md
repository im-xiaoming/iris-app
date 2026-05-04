# Bai 5

## Chay bai

```powershell
cd D:\lab_CNM\Lab8_TH\bai5
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
docker compose up -d
uvicorn app:app --reload --port 8003
```

## Link

- root: `http://127.0.0.1:8003/`
- ping: `http://127.0.0.1:8003/ping`
- docs: `http://127.0.0.1:8003/docs`
