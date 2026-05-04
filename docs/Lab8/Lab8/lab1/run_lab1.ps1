$ErrorActionPreference = "Stop"

Write-Host "Generating sample dataset..."
py -3.10 .\generate_dataset.py

Write-Host "Running H2O AutoML pipeline..."
py -3.10 .\lab1_h2o_automl.py
