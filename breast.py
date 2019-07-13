import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)



knn.score(X_train, y_train)
knn.score(X_test, y_test)
knn.score(X, y)


ypred = knn.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, ypred)