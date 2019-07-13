
#program to make fake dataset of a polynomial function 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


m = 100
#generate random number for the fake data
X = 8 * np.random.randn(m,1)
#polynomial Equation...
y = 2 * X ** 2 +3 * X + 1 + np.random.randn(m, 1)

#plot the scatter plot graph for the fake dataset

plt.scatter(X,y)
plt.axis([-3, 3, 0, 9])
plt.show()

# import Polynomial feature from sklearn

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2, include_bias = False)
X_poly = poly.fit_transform(X)

#Import the package of linear regression from sklearn API

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

#to zoom the graph for a given limit which is declared in linspace function

X_new = np.linspace(-3 ,3, 100).reshape(-1, 1)
X_new_poly = poly.fit_transform(X_new)
y_new = lin_reg.predict(X_new_poly)

#plot the graph in the form of parabola

plt.scatter(X, y)
plt.plot(X_new, y_new, c = 'r')
plt.axis([-3, 3, 0, 9])
plt.show(X, y)

#to show line coeffiecient and intercept value

lin_reg.coef_
lin_reg.intercept_

