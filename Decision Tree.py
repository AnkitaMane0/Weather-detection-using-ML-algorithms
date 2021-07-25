# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:47:39 2021

@author: MANES
"""

# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("./weather.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 3].values

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict([[6.5]])

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'purple')
plt.plot(X_grid, regressor.predict(X_grid), color = 'yellow')
plt.title('Weather Prediction')
plt.xlabel('Temperature in degree')
plt.ylabel('Precipitation')
plt.show()

