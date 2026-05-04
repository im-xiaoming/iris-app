# Lab 2 - Reproducible ML Pipeline

Lab nay tap trung vao reproducibility voi Docker va DVC, bao gom 4 yeu cau:

1. Fix random seed
2. Track environment
3. Save configs
4. Re-run pipeline

## Cau truc thu muc

```text
lab2/
|-- data/
|   `-- raw/
|-- artifacts/
|-- src/
|   |-- prepare_data.py
|   `-- train_pipeline.py
|-- .dvcignore
|-- Dockerfile
|-- docker-compose.yml
|-- dvc.yaml
|-- params.yaml
|-- README.md
`-- requirements.txt
```

## Cong nghe su dung

- Docker: dong goi moi truong chay de giam sai khac giua cac may
- DVC: dinh nghia pipeline va re-run cac stage co kiem soat
- scikit-learn: huan luyen model trong pipeline

## Yeu cau va cach dap ung

### 1. Fix random seed

File `params.yaml` luu gia tri `seed: 42`.

Trong `src/train_pipeline.py`:

- `random.seed(seed)`
- `numpy.random.seed(seed)`
- `PYTHONHASHSEED`
- `train_test_split(..., random_state=seed)`
- `RandomForestClassifier(..., random_state=seed)`

Tat ca giup pipeline cho ket qua co the lap lai.

### 2. Track environment

Pipeline luu thong tin moi truong vao `artifacts/environment.json`, bao gom:

- Python version
- Platform
- scikit-learn version
- numpy version
- pandas version
- Seed dang su dung
- Working directory

### 3. Save configs

File `params.yaml` la noi luu cau hinh trung tam:

- seed
- data path
- test size
- model hyperparameters
- artifact paths

Sau moi lan train, pipeline tao them `artifacts/config_snapshot.yaml` de luu lai cau hinh da dung o lan chay do.

### 4. Re-run pipeline

`dvc.yaml` dinh nghia 2 stages:

1. `prepare_data`
2. `train`

Chi can thay doi `params.yaml` hoac code lien quan, sau do chay:

```bash
dvc repro
```

DVC se chi re-run stage can thiet, giup pipeline co the lap lai va theo doi thay doi ro rang.

## Cach chay

### Cach 1: chay local

```bash
cd lab2
pip install -r requirements.txt
dvc repro
```

### Cach 2: chay bang Docker

Build image:

```bash
cd lab2
docker build -t lab2-reproducible-ml .
```

Run pipeline:

```bash
docker run --rm -v ${PWD}:/app lab2-reproducible-ml
```

### Cach 3: chay bang Docker Compose

```bash
cd lab2
docker compose up --build
```

## Dau ra cua pipeline

Sau khi chay thanh cong, pipeline tao:

- `data/raw/heart_disease.csv`: dataset dau vao
- `artifacts/model.joblib`: model da train
- `artifacts/metrics.json`: metrics danh gia
- `artifacts/predictions.csv`: ket qua du doan tren test set
- `artifacts/environment.json`: snapshot moi truong
- `artifacts/config_snapshot.yaml`: snapshot cau hinh

## Y nghia reproducibility trong lab nay

Lab nay dam bao tinh lap lai bang cach:

- Co dinh random seed
- Dong goi moi truong bang Docker
- Version dependencies ro rang trong `requirements.txt`
- Pipeline stages duoc mo ta bang DVC
- Config duoc luu tap trung va snapshot sau moi lan chay

## Lenh tham khao

Khoi tao DVC trong project neu can:

```bash
dvc init
```

Xem pipeline:

```bash
dvc dag
```

Chay lai tu dau:

```bash
dvc repro --force
```
