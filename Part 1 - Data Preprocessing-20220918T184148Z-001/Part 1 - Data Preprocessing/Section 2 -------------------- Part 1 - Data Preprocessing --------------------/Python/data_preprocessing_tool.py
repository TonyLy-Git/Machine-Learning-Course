# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 15:18:32 2022

@author: Tony
"""

"import libs"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"import dataset"
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# print(X)
# print(y)

"missing data"
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# print(X)

"encode categorical data"
# independent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], 
                       remainder='passthrough')
X = np.array(ct.fit_transform(X))

# print(X)

# dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# print(y)

"training and test set"
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state = 1)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

"feature scaling"

# =============================================================================
# standardisation
# works well all the time
# 
# normalisation
# recommended when normal distribution in most of features
# =============================================================================

# standardisation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# =============================================================================
# Do we have to apply feature scaling to the dummy variable in the features?
# No, bc since the features are already encoded (OneHotEncoder)
# Only apply on numerical values
# =============================================================================

X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

# print(X_train)
# print(X_test)