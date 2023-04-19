# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 17:44:03 2023

@author: lyt4
"""

# import libraries./
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
df = pd.read_csv("Position_Salaries.csv")
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# training LR model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)

# visualize results
plt.scatter(X, y, color='red')
plt.plot(X, lr.predict(X), color='blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# predict new result w/ LR
print(lr.predict([[6.5]]))

# training PR model
from sklearn.preprocessing import PolynomialFeatures
pf= PolynomialFeatures(degree = 4)
X_poly = pf.fit_transform(X)
pr = LinearRegression()
pr.fit(X_poly, y)

# visualize results
plt.scatter(X, y, color='red')
plt.plot(X, pr.predict(X_poly), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# visualize results (high res)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, pr.predict(pf.fit_transform(X_grid)), color='blue')
plt.title('Polynomial Regression (High Res)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# predict new result w/ PR
print(pr.predict(pf.fit_transform([[6.5]])))
