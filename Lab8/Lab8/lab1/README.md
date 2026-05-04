# Lab 1 - AutoML Pipeline with H2O.ai

Lab nay hoan thanh 4 yeu cau chinh:

1. Automated feature engineering
2. Model selection bang H2O AutoML
3. Evaluation tren tap test
4. Export model

## Cau truc

```text
lab1/
|-- data/
|   `-- breast_cancer_with_missing.csv
|-- artifacts/
|   |-- engineered_dataset.csv
|   |-- evaluation_report.json
|   |-- leaderboard.csv
|   `-- models/
|-- generate_dataset.py
|-- lab1_h2o_automl.py
|-- run_lab1.ps1
`-- requirements.txt
```

## Cach chay

```bash
cd lab1
py -3.10 -m pip install -r requirements.txt
py -3.10 generate_dataset.py
py -3.10 lab1_h2o_automl.py
```

Hoac chay mot lan:

```powershell
cd lab1
./run_lab1.ps1
```

## Noi dung pipeline

### 1. Automated feature engineering

Script `lab1_h2o_automl.py` tu dong thuc hien:

- Dien gia tri thieu cho bien so bang median
- Tao co missing indicator cho cot co gia tri thieu
- Tao interaction features cho nhom numeric co variance cao
- Tao quantile bins cho mot so cot numeric quan trong
- Xu ly cot categorical va tao top-5 category flags neu co

### 2. Model selection

H2O AutoML duoc cau hinh de thu nhieu thuat toan:

- GBM
- XGBoost
- DRF
- GLM
- StackedEnsemble

Model tot nhat duoc chon dua tren chi so `AUC`.

### 3. Evaluation

Pipeline chia du lieu thanh:

- 70% training
- 15% validation
- 15% test

Bao cao `artifacts/evaluation_report.json` luu cac metric:

- AUC
- Accuracy
- F1
- Logloss

### 4. Export model

Pipeline se export:

- Binary model cua leader vao `artifacts/models/`
- MOJO neu H2O ho tro trong moi truong hien tai
- Leaderboard vao `artifacts/leaderboard.csv`

## Ket qua dau ra

Sau khi chay thanh cong, ban se nhan duoc:

- Dataset da feature engineer
- Leaderboard cac model
- File JSON tong hop ket qua
- Thu muc model export de deploy hoac nap lai sau nay

## Ghi chu

- Lab su dung dataset phan loai ung thu vu tu `scikit-learn`
- `generate_dataset.py` co chen them mot it missing values de minh hoa buoc feature engineering tu dong
- Can Java de H2O khoi dong local cluster
- Neu may co nhieu version Python, nen uu tien Python 3.10 de tranh loi tuong thich voi `h2o`
