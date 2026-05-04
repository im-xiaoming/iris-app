# Lab 2: Data Versioning voi DVC

## Muc tieu

Quan ly version du lieu va tai hien pipeline xu ly Titanic dataset bang DVC.

## Da trien khai

- Track dataset version.
- Push len remote storage.
- Reproduce pipeline.
- So sanh version.

## Cai dat

```powershell
cd "C:\Nam 4\CNM\lab7"
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r .\lab2_dvc\requirements.txt
```

## Khoi tao DVC

```powershell
cd .\lab2_dvc
git init
dvc init
dvc add .\data\raw\titanic.csv
git add .
git commit -m "Init Titanic DVC lab"
```

## Cau hinh remote

Demo voi remote local:

```powershell
mkdir .\dvc_remote
dvc remote add -d localremote .\dvc_remote
dvc push
```

## Chay pipeline

```powershell
dvc repro
python .\src\compare_versions.py
```
