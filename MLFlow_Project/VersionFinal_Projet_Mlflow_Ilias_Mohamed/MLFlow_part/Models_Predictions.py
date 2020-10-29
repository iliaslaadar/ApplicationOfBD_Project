#!/usr/bin/env python
# coding: utf-8

# # Ilias LAADAR - Mohamed ABDELAZIZ - Applications of Big Data

# ## Models Predictions

# In[1]:


import os
import warnings
import sys

import pandas as pd


"""
    Predictions of the model

    Args:
        ***_model: The model we want to use
        X_test (DataFrame): The values to test the models

    Returns:
        ***_pred (DataFrame): Model predictions
"""


# In[2]:


def XGB_pred(xgb_model, X_test):
    xgb_pred = xgb_model.predict(X_test)
    return xgb_pred


# In[3]:


def RF_pred(rf_model, X_test):
    rf_pred = rf_model.predict(X_test)
    return rf_pred


# In[4]:


def GBC_pred(gbc_model, X_test):
    gbc_pred = gbc_model.predict(X_test)
    return gbc_pred

