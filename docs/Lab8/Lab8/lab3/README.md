# Lab 3 - CI/CD cho ML

Lab nay tap trung vao automation cho machine learning voi Jenkins va GitHub Actions.

## Muc tieu

1. Build pipeline
2. Test model
3. Deploy tu dong
4. Rollback

## Cau truc thu muc

```text
lab3/
|-- .github/
|   `-- workflows/
|       `-- ml-cicd.yml
|-- app/
|   `-- main.py
|-- data/
|   `-- raw/
|-- deployment/
|   `-- registry/
|-- models/
|-- reports/
|-- src/
|   |-- train_model.py
|   |-- validate_model.py
|   |-- deploy_model.py
|   `-- rollback_model.py
|-- tests/
|   `-- test_training_outputs.py
|-- Jenkinsfile
|-- params.yaml
|-- README.md
`-- requirements.txt
```

## Yeu cau va cach dap ung

### 1. Build pipeline

Build pipeline duoc thuc hien trong `src/train_model.py`:

- Tao dataset classification tu `scikit-learn`
- Chia train/test voi seed co dinh
- Train `RandomForestClassifier`
- Luu model vao `models/model.joblib`
- Luu metrics vao `reports/metrics.json`

Ca Jenkins va GitHub Actions deu goi buoc nay trong stage build.

### 2. Test model

Buoc test gom 2 lop:

- `src/validate_model.py` kiem tra metrics dat threshold deploy
- `tests/test_training_outputs.py` dung `pytest` de assert file output va chat luong model

Neu model khong dat nguong, pipeline se dung truoc khi deploy.

### 3. Deploy tu dong

Buoc deploy duoc thuc hien boi `src/deploy_model.py`:

- Tao version model moi trong `deployment/registry/`
- Cap nhat `deployment/current` tro toi version moi nhat
- Giu lai version truoc do qua `deployment/previous`

Trong CI/CD:

- Jenkins chi deploy khi branch la `main`
- GitHub Actions chi deploy khi push len `main`

### 4. Rollback

Rollback duoc thuc hien boi `src/rollback_model.py`:

- Neu deploy that bai hoac pipeline failure sau deploy, `current` se duoc tro lai `previous`
- Jenkins dung `post { failure { ... } }`
- GitHub Actions dung `if: failure()`

## Jenkins Pipeline

File `Jenkinsfile` mo ta cac stage:

1. Checkout
2. Setup environment
3. Build Pipeline
4. Test Model
5. Deploy

Va co `post failure` de rollback tu dong.

## GitHub Actions Workflow

File `.github/workflows/ml-cicd.yml` tu dong:

- Checkout code
- Cai dependency
- Train model
- Validate va test
- Deploy neu hop le
- Rollback neu failure

## Cach su dung

### Chay tay local

```bash
cd lab3
pip install -r requirements.txt
python src/train_model.py
python src/validate_model.py
pytest tests
python src/deploy_model.py
```

### Rollback thu cong

```bash
cd lab3
python src/rollback_model.py
```

### Chay API phuc vu model

```bash
cd lab3
uvicorn app.main:app --reload
```

API co:

- `GET /health`
- `POST /predict`

## Dau ra sau khi pipeline chay

- `models/model.joblib`: model moi nhat sau build
- `reports/metrics.json`: ket qua danh gia
- `reports/validation.json`: ket qua validate truoc deploy
- `deployment/registry/`: lich su version da deploy
- `deployment/current`: version dang phuc vu
- `deployment/previous`: version dung de rollback

## Ghi chu

- Lab nay mo phong CI/CD cho ML theo huong de hoc va demo
- Trong thuc te, deploy co the day artifact len server, S3, model registry, Kubernetes, hoac cloud endpoint
- Co the mo rong them notification, canary deploy, hoac approval gate
