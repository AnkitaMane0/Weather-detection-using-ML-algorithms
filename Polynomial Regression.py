# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:38:08 2021

@author: MANES
"""

# Polynomial Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("./weather.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'Pink')
plt.plot(X, lin_reg.predict(X), color = 'Black')
plt.title('Weather Prediction')
plt.xlabel('Temperature in degree')
plt.ylabel('Precipitation')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'pink')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'black')
plt.title('Weather Prediction')
plt.xlabel('Temperature in degree')
plt.ylabel('Precipitation')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'pink')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'black')
plt.title('Weather Prediction')
plt.xlabel('Temperature in degree')
plt.ylabel('Precipitation')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
