dvc stage add -n preprocess \
-p preprocess.input,preprocess.output \
-d ml/preprocess.py -d data/raw/iris.csv \
-o data/preprocessed/iris_preprocessed.csv \
python ml/preprocess.py

dvc stage add -n train \
-p train.data_path,train.model_path,train.scaler_path,train.parameters \
-d ml/train.py -d data/preprocessed/iris_preprocessed.csv \
-o models/iris_model.pkl -o models/iris_scaler.pkl \
--force \
python ml/train.py

dvc repro