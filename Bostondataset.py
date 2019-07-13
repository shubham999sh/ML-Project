import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_boston

dataset = load_boston()
X = dataset.data
X = dataset.data[:, 1]
#y = dataset.iloc[:,-1]
y = dataset.target

#X = X.reshape(-1,1)


# line ko plot karne ke liye
plt.scatter(X,y)
#plt.plot(X, lin_reg.predict(X), c = 'r')
plt.show()

# Import Linear Regression from sklearn
from sklearn.linear_model import LinearRegression
lin_reg  = LinearRegression()
lin_reg.fit(X,y)

# to check score

lin_reg.score(X,y)


plt.scatter(X,y)
plt.plot(X, lin_reg.predict(X), c = 'r')
plt.show()

#predict the value of lin_reg class values

lin_reg.predict([[20]])
lin_reg.predict([[26]])


lin_reg.coef_
lin_reg.intercept_


y_pred = lin_reg.predict(X)

