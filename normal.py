import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_excel('Desktop/blood.xlsx')
X = dataset.iloc[2:, 1].values
y = dataset.iloc[2:,-1].values
X = X.reshape(-1,1)

# Import Linear Regression from sklearn
from sklearn.linear_model import LinearRegression
lin_reg  = LinearRegression()
lin_reg.fit(X,y)

# to check score
lin_reg.score(X,y)

# line ko plot karne ke liye
plt.scatter(X,y)
plt.plot(X, lin_reg.predict(X), c = 'r')
plt.show()

lin_reg.predict([[20]])
lin_reg.predict([[26]])

lin_reg.coef_
lin_reg.intercept_

y_pred = lin_reg.predict(X)



