# Lab 5 - Real-time Prediction System

Lab nay xay dung he thong du doan realtime voi FastAPI va Redis.

## Muc tieu

1. API serving
2. Low latency
3. Caching
4. Logging

## Cau truc thu muc

```text
lab5/
|-- data/
|   `-- raw/
|-- logs/
|-- models/
|-- reports/
|-- src/
|   |-- cache.py
|   |-- logger_config.py
|   |-- main.py
|   `-- train_realtime_model.py
|-- docker-compose.yml
|-- params.yaml
|-- README.md
`-- requirements.txt
```

## Yeu cau va cach dap ung

### 1. API serving

File `src/main.py` dung FastAPI de phuc vu du doan realtime.

Co 2 endpoint:

- `GET /health`: kiem tra tinh trang service
- `POST /predict`: nhan input va tra ve du doan

### 2. Low latency

He thong duoc thiet ke toi uu cho latency thap:

- Model duoc load 1 lan khi service khoi dong
- Khong retrain khi co request
- Dung `time.perf_counter()` de theo doi latency moi request
- Input validation bang Pydantic de giam loi runtime
- Co tham so `timeout_ms` trong `params.yaml` de mo rong quan ly SLA

### 3. Caching

File `src/cache.py` tao lop `RedisCache`:

- Dung Redis de cache ket qua prediction
- Dung SHA-256 fingerprint cua feature vector lam key
- Co TTL trong `params.yaml`
- Neu cung input lap lai, he thong doc ket qua tu Redis thay vi predict lai

Dieu nay giup giam latency va tang throughput cho request lap lai.

### 4. Logging

File `src/logger_config.py` cau hinh logging ra:

- Console
- File `logs/prediction_service.log`

Moi request se duoc log:

- request id
- nguon ket qua: `cache` hoac `model`
- latency
- prediction result
- loi cache hoac loi predict neu co

## Quy trinh hoat dong

1. Train model bang `src/train_realtime_model.py`
2. Service FastAPI load model khi startup
3. Request toi `/predict`
4. Kiem tra cache trong Redis
5. Neu co cache thi tra ket qua ngay
6. Neu khong co thi model predict, sau do luu vao cache
7. Logging thong tin request va latency

## Cach chay

### Chay local

```bash
cd lab5
pip install -r requirements.txt
python src/train_realtime_model.py
uvicorn src.main:app --reload
```

### Chay voi Redis bang Docker Compose

```bash
cd lab5
docker compose up
```

## Vi du request

```json
{
  "request_id": "req_001",
  "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.0787, 1.095, 0.9053, 8.589, 153.4, 0.0064, 0.049, 0.0537, 0.0159, 0.03, 0.0062, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
}
```

## Vi du response

```json
{
  "request_id": "req_001",
  "prediction": "malignant",
  "cache_hit": false,
  "latency_ms": 3.421
}
```

## Dau ra sau khi chay

- `models/realtime_model.joblib`: model phuc vu realtime
- `reports/training_metrics.json`: metrics sau huan luyen
- `logs/prediction_service.log`: log request va latency

## Ghi chu

- Redis dong vai tro cache layer de giam thoi gian suy luan cho input lap lai
- FastAPI phu hop cho service ML realtime nho hieu nang tot va schema ro rang
- Co the mo rong them rate limiting, async inference, monitoring, va load balancing
