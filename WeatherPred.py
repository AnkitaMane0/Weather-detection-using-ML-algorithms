# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:47:00 2021

@author: MANES
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #generating graphs

#Importing DataSet
dataset = pd.read_csv("./weather.csv")
temp=dataset['temperaturemin']
precipitation=dataset['precipitation']

dataset.head(10)

x = np.array(temp).reshape(-1, 1) # function name says it : reshape the array
y = np.array(precipitation)

#Splitting the data into Train and Test
#from sklearn.cross_validation import train_test_split in earlier version
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split( x, y, test_size=1/3, random_state=0 )

#Fitting simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit( xtrain, ytrain )

regressor.coef_ , regressor.intercept_# y = mx + c , m is coef , c is intercept

actualValue = ytrain
predictedValue = regressor.predict(xtrain)
xtrain[0], actualValue[0] , predictedValue[0]

regressor.coef_ * xtrain[0] + regressor.intercept_ # y = mx + c

np.sqrt ( sum( abs( actualValue**2 - predictedValue**2 ) ) ) / len( xtrain ) # RMSE
# Root Mean Square Error

#Visualizing the training Test Results
# Actual values
plt.scatter(xtrain, ytrain, color='green') # x = xtrain , y = ytrain
#Predicted values
prediction = regressor.predict(xtrain)
plt.plot(xtrain, prediction , color = 'black') # y = prediction
plt.title ("Prediction for Training Dataset")
plt.xlabel("Temperature in degree"), plt.ylabel("Precipitation")
plt.show()

#Visualizing the Test Results
plt.scatter(xtest, ytest, color= 'green')
plt.plot(xtrain, regressor.predict(xtrain), color = 'black')
plt.title ("Training Dataset")
plt.xlabel("Tempertaure in degree"), plt.ylabel("Precipitation")
plt.show()

d=dataset['avgwindspeed'].value_counts()

d.plot(kind='bar')



