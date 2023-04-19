# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 18:53:33 2023

@author: lyt4
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
df = pd.read_csv("Position_Salaries.csv")
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# training decision tree regression model
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X, y)

# predicting new result
print(dtr.predict([[6.5]]))

# visualizing results (high res)