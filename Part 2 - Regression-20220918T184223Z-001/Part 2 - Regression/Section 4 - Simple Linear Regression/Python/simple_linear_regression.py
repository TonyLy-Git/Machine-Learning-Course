# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 17:15:38 2022

@author: Tony
"""

"""
y = b0 + b1*x
dependent variable(DV) = constant + coefficient * independent variable(IV)
linear regression is a trend line (best fit)
"""

#import libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

#regression model on training set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

#predict test set results
y_pred = lr.predict(X_test)

#training set results
plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, lr.predict(X_train), color = 'red')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

#test set results
plt.scatter(X_test, y_test, color = 'green')

#single prediction
print(lr.predict([[12]]))

#linear regression equation
print(lr.coef_)
print(lr.intercept_)
# Salary= 26816.19 + 9345.94 * YearsExperience
