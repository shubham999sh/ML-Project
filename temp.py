import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_excel("Desktop/blood.xlsx")
X = dataset.iloc[2:,1].values
y = dataset.iloc[2:,-1].valuesplt.scatter(X_train, y_train)
plt.plot(X_train,lin_reg.predict(Xtrain),c = "r")
plt.show()

X = X.reshape(-1,1)


plt.scatter(X,y)
plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


