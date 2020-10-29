#!/usr/bin/env python
# coding: utf-8

# # Ilias LAADAR - Mohamed ABDELAZIZ - Applications of Big Data

# ## Models Evaluation

# In[1]:


import os
import warnings
import sys

import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[2]:


def evaluation_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    f1score = f1_score(actual, pred, average='micro')
    precision = precision_score(actual, pred, average='weighted')
    recall = recall_score(actual, pred, average='micro')
    
    return accuracy, f1score, precision, recall
