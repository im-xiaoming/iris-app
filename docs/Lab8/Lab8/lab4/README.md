# Lab 4 - A/B Testing cho Model

Lab nay mo phong viec so sanh 2 model trong moi truong deployment bang Flask va Kubernetes.

## Muc tieu

1. Deploy 2 models
2. Split traffic
3. Compare metrics
4. Select best

## Cau truc thu muc

```text
lab4/
|-- data/
|   |-- evaluation/
|   `-- raw/
|-- deployment/
|   `-- production/
|-- k8s/
|   |-- ab-router-deployment.yaml
|   |-- ab-router-service.yaml
|   |-- model-a-deployment.yaml
|   `-- model-b-deployment.yaml
|-- models/
|-- reports/
|-- src/
|   |-- ab_router.py
|   |-- evaluate_ab_test.py
|   |-- promote_best_model.py
|   `-- train_models.py
|-- Dockerfile
|-- params.yaml
|-- README.md
`-- requirements.txt
```

## Yeu cau va cach dap ung

### 1. Deploy 2 models

File `src/train_models.py` train 2 model khac nhau:

- Model A: `RandomForestClassifier`
- Model B: `GradientBoostingClassifier`

Hai model duoc luu thanh:

- `models/model_a.joblib`
- `models/model_b.joblib`

Trong thu muc `k8s/` da co manifest mo phong deployment cho:

- `model-a`
- `model-b`
- `ab-router`

### 2. Split traffic

File `src/ab_router.py` dung Flask lam router:

- Nhan request prediction
- Dinh danh user theo `user_id`
- Hash `user_id` de chia traffic on dinh
- Route 50/50 theo cau hinh trong `params.yaml`

Voi cung mot `user_id`, user se luon vao cung mot nhom A hoac B.

### 3. Compare metrics

Moi request co the gui kem `actual_label` de phuc vu evaluation.

Router se log vao:

- `data/evaluation/ab_test_requests.csv`

Sau do script `src/evaluate_ab_test.py` se:

- Tinh accuracy cho model A va B
- Tinh F1 cho lop `malignant`
- So sanh sample count moi nhom
- Ghi ket qua vao `reports/winner.json`

### 4. Select best

Script `src/promote_best_model.py` doc `reports/winner.json` va:

- Chon model co ket qua tot hon
- Copy model chien thang vao `deployment/production/best_model.joblib`

Neu hai model hoa nhau, script se dung de tranh promote sai.

## Flask API

Chay router:

```bash
cd lab4
python src/train_models.py
python src/ab_router.py
```

Endpoints:

- `GET /health`
- `GET /config`
- `POST /predict`

Vi du request:

```json
{
  "user_id": "user_001",
  "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.0787, 1.095, 0.9053, 8.589, 153.4, 0.0064, 0.049, 0.0537, 0.0159, 0.03, 0.0062, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189],
  "actual_label": "malignant"
}
```

## Kubernetes

Cac file trong `k8s/` minh hoa cach trien khai:

- 2 deployment cho 2 model versions
- 1 deployment + service cho A/B router

Co the apply nhu sau:

```bash
kubectl apply -f k8s/model-a-deployment.yaml
kubectl apply -f k8s/model-b-deployment.yaml
kubectl apply -f k8s/ab-router-deployment.yaml
kubectl apply -f k8s/ab-router-service.yaml
```

## Cach danh gia va chon model tot nhat

Sau khi co du lieu tu traffic thuc te hoac traffic mo phong:

```bash
cd lab4
python src/evaluate_ab_test.py
python src/promote_best_model.py
```

Ket qua:

- `reports/ab_metrics.json`: metrics offline cua 2 model sau train
- `reports/winner.json`: ket qua A/B test va model duoc chon
- `deployment/production/best_model.joblib`: model duoc promote

## Ghi chu

- Lab nay mo phong logic A/B testing cho model trong MLOps
- Trong thuc te, metric online co the la CTR, conversion, latency, reject rate, business KPI
- Co the mo rong them dashboard, Prometheus, Grafana, hay service mesh de split traffic nang cao
