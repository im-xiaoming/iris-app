# Bai 4

## Chay local

```powershell
cd D:\lab_CNM\Lab8_TH\bai4
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
python app.py
```

- root: `http://127.0.0.1:8001/`
- stats: `http://127.0.0.1:8001/stats`
- best: `http://127.0.0.1:8001/best`

## K8s

```powershell
docker build -t bai4:latest .
kubectl apply -f k8s/app.yaml
kubectl port-forward service/ab-api 8001:80
```

- can bat Kubernetes truoc khi chay `kubectl`
- neu dang dung minikube thi chay them `minikube image load bai4:latest`

## Link

- API: `http://127.0.0.1:8001`
