# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:09:56 2023

@author: TLy
"""

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
data = pd.read_csv('Social_Network_Ads.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# split data into train/test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
"""
print(X_train)
print(y_train)
print(X_test)
print(y_test)
"""

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""
print(X_train)
print(X_test)
"""

# train logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train)

# predict new result
"""
print(lr.predict(sc.transform([[30,87000]])))
"""

# predict test set result
y_pred = lr.predict(X_test)
"""
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
"""

# maing confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
"""
print(cm)
predicted not to buy (class 0): 65 correct, 8 incorrect
predicted to buy (class 1): 24 correct, 3 incorrect
"""

score = accuracy_score(y_test, y_pred)
"""
print(score)
89% of correct predictions in test set
"""

# visualize train set
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop=X_set[:,0].max() + 10, step=0.25), 
                     np.arange(start=X_set[:,1].min() - 1000, stop=X_set[:,1].max() + 1000, step=.25))
plt.contourf(X1, X2, lr.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('black', 'white'))(i), label = j)
plt.title('LR (Train)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# visualize test set
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop=X_set[:,0].max() + 10, step=0.25), 
                     np.arange(start=X_set[:,1].min() - 1000, stop=X_set[:,1].max() + 1000, step=.25))
plt.contourf(X1, X2, lr.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('black', 'white'))(i), label = j)
plt.title('LR (Test)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()