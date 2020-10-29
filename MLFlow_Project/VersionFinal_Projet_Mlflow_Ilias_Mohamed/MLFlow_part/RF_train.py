#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import warnings
import sys

import numpy as np

from Data_Preparation import data_preparation
from Feature_Engineering import feature_engineering
from Models_Training import RF_model
from Models_Predictions import RF_pred
from Models_Evaluation import evaluation_metrics


from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # Data Preparation
    df_train = data_preparation()
    
    # Feature Engineering
    X_train, X_test, y_train, y_test = feature_engineering(df_train)
    
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    

    with mlflow.start_run():
        
        # Model Training
        rf_model = RF_model(n_estimators, X_train, y_train)
        
        # Model Prediction
        rf_pred = RF_pred(rf_model, X_test)
        
        # Metrics Evaluation
        (accuracy, f1score, precision, recall) = evaluation_metrics(y_test, rf_pred)

        # Log parameter, metrics, and model to MLflow
        mlflow.log_param("N_estimators", n_estimators)
        mlflow.log_param("Learning_rate", learning_rate)
        
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F1_score", f1score)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            mlflow.sklearn.log_model(rf_model, "rf_model", registered_model_name="RandomForest model")
        else:
            mlflow.sklearn.log_model(rf_model, "rf_model")

