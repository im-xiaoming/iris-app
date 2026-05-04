dvc stage add -n preprocess
-p preprocess.input,preprocess.output
-d src/preprocess.py -d data/raw/iris.csv
-o data/preprocessed/iris_preprocessed.csv
python3 src/preprocess.py

dvc stage add -n train
-p train.data_path,train.model_path,train.parameters
-d src/train.py -d data/preprocessed/iris_preprocessed.csv
-o models/iris_model.pkl -o models/scaler.pkl
python3 src/train.py

dvc repro