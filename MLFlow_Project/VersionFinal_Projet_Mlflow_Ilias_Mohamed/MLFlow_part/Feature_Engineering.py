#!/usr/bin/env python
# coding: utf-8

# # Ilias LAADAR - Mohamed ABDELAZIZ - Applications of Big Data

# ## Feature Engineering 

# In[1]:


import os
import warnings
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


"""
    Feature engineering

    Args:
        df_train (DataFrame): Prepared data

    Returns:
        X_train (DataFrame): The values to train the models
        y_train (DataFrame): The labels to train the models
        X_test (DataFrame): The values to test the models
        y_test (DataFrame): The labels to test the models
"""


# In[2]:


def feature_engineering(df_train):
    df_train = pd.get_dummies(df_train)
    
    y = df_train['TARGET']
    df_train = df_train.drop(['TARGET'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size = 0.2, random_state = 42)
    
    return X_train, X_test, y_train, y_test

