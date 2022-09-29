# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 14:41:07 2022

@author: lyt4
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("50_Startups.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)