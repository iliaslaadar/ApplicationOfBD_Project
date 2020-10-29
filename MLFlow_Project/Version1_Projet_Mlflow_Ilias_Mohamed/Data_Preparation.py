#!/usr/bin/env python
# coding: utf-8

# # Ilias LAADAR - Mohamed ABDELAZIZ - Applications of Big Data

# ## Data Preparation

# In[1]:


import os
import warnings
import sys

import pandas as pd


# In[2]:


def data_preparation():
    df_train = pd.read_csv("application_train.csv")
    
    # Replace missing values by the mean of the feature
    df_str = df_train.select_dtypes(include='object').fillna('Unknown')
    df_train = df_train.select_dtypes(exclude='object')
    df_train = df_train.fillna(df_train.mean())
    
    df_train = df_train.join(df_str)
    
    return df_train

