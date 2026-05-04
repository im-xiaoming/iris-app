from sklearn.metrics import accuracy_score, recall_score
from dotenv import load_dotenv
import yaml
import os
import mlflow
import optuna
import sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
import joblib


load_dotenv()

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')

params = yaml.safe_load(open('params.yaml'))['train']


def hyperparameter_tuning(X_train, X_test, y_train, y_test, parameters):

    def objective(trial):
        # Setting nested=True will create a child run under the parent run.
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:
            rf_max_depth = trial.suggest_int("rf_max_depth", parameters['rf_max_depth']['min'], parameters['rf_max_depth']['max'])
            
            rf_n_estimators = trial.suggest_int("rf_n_estimators", parameters['rf_n_estimators']['min'], parameters['rf_n_estimators']['max'], step=10)
            
            rf_max_features = trial.suggest_float("rf_max_features", parameters['rf_max_features']['min'], parameters['rf_max_features']['max'])
            
            params = {
                "max_depth": rf_max_depth,
                "n_estimators": rf_n_estimators,
                "max_features": rf_max_features,
            }
            # Log current trial's parameters
            mlflow.log_params(params)

            model = sklearn.ensemble.RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            
            error = sklearn.metrics.mean_squared_error(y_test, y_pred)
            acc_score = accuracy_score(y_test, y_pred)
            rc_score = recall_score(y_test, y_pred, average='weighted')
            # Log current trial's error metric
            mlflow.log_metrics({
                "error": error,
                'accuracy': acc_score,
                'recall': rc_score,
            })
            
            signature = infer_signature(X_test, y_pred)
            
            # Log the model file
            mlflow.sklearn.log_model(model, name="model", signature=signature, registered_model_name=f"sk-learn-random-forest-model_{trial.number}") 
            # Make it easy to retrieve the best-performing child run later
            trial.set_user_attr("run_id", child_run.info.run_id)
            trial.set_user_attr("model", model)
            
            return rc_score



    # Create a parent run that contains all child runs for different trials
    with mlflow.start_run(run_name="study") as run:
        # Log the experiment settings
        n_trials = 1
        mlflow.log_param("n_trials", n_trials)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        # Log the best trial and its run ID
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metrics({"best_recall": study.best_value})
        
        best_run_id = study.best_trial.user_attrs.get("run_id")
        best_model = study.best_trial.user_attrs.get("model")
        
        mlflow.log_param("best_child_run_id", best_run_id)
        
        return best_model
    
     
def train(data_path, model_path, scaler_path, parameters):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("iris_classification_experiment_v3")
    
    
    df = pd.read_csv(data_path)
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    best_model = hyperparameter_tuning(X_train, X_test, y_train, y_test, parameters)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print("Training completed. Best model saved to:", model_path)
    
    
if __name__ == '__main__':
    data_path = params['data_path']
    model_path = params['model_path']
    scaler_path = params['scaler_path']
    parameters = params['parameters']
    
    train(data_path, model_path, scaler_path, parameters)