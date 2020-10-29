#!/usr/bin/env python
# coding: utf-8

# # Ilias LAADAR - Mohamed ABDELAZIZ - Applications of Big Data

# ## Models Training

# In[1]:


import os
import warnings
import sys

import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


"""
    Creation and training of the model

    Args:
        n_estimators (int): The number of estimaros
        learning_rate (float): The value we want to use pour the learning rate
        X_train (DataFrame): The values to train the models
        y_train (DataFrame): The labels to train the models

    Returns:
        ***_model (DataFrame): The model we want to use
"""


# In[2]:


def XGB_model(n_estimators, learning_rate, X_train, y_train):
    xgb_model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    return xgb_model


# In[3]:


def RF_model(n_estimators, X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return rf_model


# In[4]:


def GBC_model(n_estimators, learning_rate, X_train, y_train):
    gbc_model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    gbc_model.fit(X_train, y_train)
    
    return gbc_model

