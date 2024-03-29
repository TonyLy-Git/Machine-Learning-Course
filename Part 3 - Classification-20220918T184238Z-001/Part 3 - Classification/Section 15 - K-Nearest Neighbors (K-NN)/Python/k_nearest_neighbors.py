# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:15:46 2023

@author: lyt4
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

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# train K-NN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train, y_train)

# predict new result
print(knn.predict(sc.transform([[30,87000]])))

# predict test set results
y_pred = knn.predict(X_test)
"""
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
"""
# make confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

print(cm)
"""
predicted not to buy (class 0): 64 correct, 3 incorrect
predicted to buy (class 1): 29 correct, 4 incorrect
"""

score = accuracy_score(y_test, y_pred)

print(score)
"""
93% of correct predictions in test set
"""

# visualize train set
from matplotlib.colors import ListedColormap
"""
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop=X_set[:,0].max() + 10, step=1), 
                     np.arange(start=X_set[:,1].min() - 1000, stop=X_set[:,1].max() + 1000, step=1))
plt.contourf(X1, X2, knn.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('black', 'white'))(i), label = j)
plt.title('K-NN (Train)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
"""

# visuaize test set
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop=X_set[:,0].max() + 10, step=1), 
                     np.arange(start=X_set[:,1].min() - 1000, stop=X_set[:,1].max() + 1000, step=1))
plt.contourf(X1, X2, knn.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('black', 'white'))(i), label = j)
plt.title('K-NN (Test)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
