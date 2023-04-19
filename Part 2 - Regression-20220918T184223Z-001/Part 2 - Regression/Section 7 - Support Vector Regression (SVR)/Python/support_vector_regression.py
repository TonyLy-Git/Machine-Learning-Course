# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:32:58 2023

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
y = y.reshape(len(y), 1)
#print(X)
#print(y)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
#print(X)
#print(y)

# traning SVR model
from sklearn.svm import SVR
svr = SVR(kernel='rbf') #model
svr.fit(X, y)

# predict new result
sc_y.inverse_transform(svr.predict(sc_X.transform([[6.5]])).reshape(-1, 1))

# visualizing results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(svr.predict(X).reshape(-1, 1)), color='blue')
plt.title('SVR')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# visualizing results (high res)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(X_grid, sc_y.inverse_transform(svr.predict(sc_X.transform(X_grid)).reshape(-1, 1)), color='blue')
plt.title('SVR (High Res)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
