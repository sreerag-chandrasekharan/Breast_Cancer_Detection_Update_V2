import os, json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import yaml
import mlflow
import mlflow.sklearn

#--- Load the dataset
data = pd.read_csv('C:/Users/Sreerag/Documents/ML_chellange/Breast_Cancer_Detection_Update_V2/data/final/data_final.csv')

#--- Split the dataset into features and target variable
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

#--- Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#--- Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#--- import the parameters from params.yaml
with open('C:/Users/Sreerag/Documents/ML_chellange/Breast_Cancer_Detection_Update_V2/params.yaml', 'r') as file:
    params = yaml.safe_load(file)["train"]

#--- model pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        penalty= params['penalty'],
        C= params['C'], 
        solver='liblinear', 
        class_weight=params['class_weight'], 
        max_iter=params['max_iter'], 
        random_state=42))
])  

#--- MLflow setup
mlflow.set_experiment('breast_cancer_detection')
with mlflow.start_run():
    #--- Train the model
    pipe.fit(X_train_scaled, y_train)
    predict = pipe.predict(X_test_scaled)
    prob = pipe.predict_proba(X_test_scaled)[:, 1]
    #--- Log the model
    metrics = {
        'roc_auc': float(roc_auc_score(y_test, prob)),
        'f1': float(f1_score(y_test, predict)),
        'accuracy': float(accuracy_score(y_test, predict))
    }
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(pipe, "model")
    mlflow.log_params(params)

#----save the model

os.makedirs("models", exist_ok=True)
model_path = "Model/model.pkl"
mlflow.sklearn.save_model(pipe, "Model/mlflow_model")
import joblib; joblib.dump(pipe, model_path)

#------- also write metrics for DVC
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

#---- log artifacts
    mlflow.log_artifact("metrics.json")
    mlflow.log_artifact(model_path)

    print("Run ID:", run.info.run_id)