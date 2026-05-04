# Lab 7: Automated Training Pipeline voi Kubeflow

## Muc tieu

Tu dong hoa quy trinh training model bang Kubeflow:

- Trigger training.
- Data preprocessing.
- Training.
- Evaluation.
- Save model.

## Cau truc

```text
lab7_kubeflow/
|-- artifacts/
|-- data/
|   `-- student_performance.csv
|-- pipeline/
|   `-- automated_training_pipeline.py
|-- README.md
|-- requirements.txt
`-- src/
    |-- compile_pipeline.py
    `-- run_local_pipeline.py
```

## Cong nghe

- Kubeflow Pipelines SDK
- scikit-learn
- pandas

## Cai dat

```powershell
cd "C:\Nam 4\CNM\lab7"
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r .\lab7_kubeflow\requirements.txt
```

## Trigger training local

```powershell
python .\lab7_kubeflow\src\run_local_pipeline.py
```

Script nay mo phong qua trinh trigger training cuc bo va tao:

- `artifacts/train.csv`
- `artifacts/test.csv`
- `artifacts/model.pkl`
- `artifacts/evaluation.json`

## Compile Kubeflow pipeline

```powershell
python .\lab7_kubeflow\src\compile_pipeline.py
```

Lenh tren se tao file:

- `lab7_kubeflow/artifacts/automated_training_pipeline.yaml`

File YAML co the import vao Kubeflow Pipelines UI de chay pipeline.

## Cac buoc trong pipeline

1. `trigger_training`: Khoi tao run id va trigger metadata.
2. `preprocess_data`: Lam sach du lieu, chia train/test.
3. `train_model`: Train mo hinh classification.
4. `evaluate_model`: Tinh cac metric danh gia.
5. `save_model`: Luu model metadata va artifact.

## Dau vao va dau ra

- Dau vao: dataset `student_performance.csv`
- Dau ra: model logistic regression duoc luu va metric accuracy/f1/precision/recall
