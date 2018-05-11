# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:43:38 2018

@author: lukasz.dymanski
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 14:47:22 2018

@author: lukasz.dymanski
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re


# Read training Dataset
dataset = pd.read_csv('train.csv')
y = dataset.iloc[:, 1].values
dataset = dataset.drop(['PassengerId','Survived','Ticket', 'Cabin', 'Embarked'],1)
#dataset['Embarked'] = dataset['Embarked'].fillna("S")

# Names preprocessing
Names = dataset['Name'].str.split(', ').str[1].str.split('.').str[0]
Titles = ['Mr', 'Mrs', 'Miss', 'Master', 'Dr']
Names = Names.apply(lambda i: i if i in Titles else 'None')
dataset['Name'] = Names

X = dataset.iloc[:,:].values


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer_numeric= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_numeric = imputer_numeric.fit(X[:, [3]])
X[:, [3]] = imputer_numeric.transform(X[:, [3]])



# Encoding categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
labelencoder_X2 = LabelEncoder()
labelencoder_X3 = LabelEncoder()
labelencoder_X3 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X1.fit_transform(X[:, 2])
X[:, 7] = labelencoder_X2.fit_transform(X[:, 7])
onehotencoder = OneHotEncoder(categorical_features = [1,2])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
#X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Opt features
X_train = X_train[:, [0, 1, 2, 3 , 6, 7, 8, 9, 10, 11, 12]]
X_test = X_test[:, [0, 1, 2, 3 , 6, 7, 8, 9, 10, 11, 12]]

# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression()
classifier_LR.fit(X_train, y_train)
LR_score = classifier_LR.score(X_test, y_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier_kNN = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
classifier_kNN.fit(X_train, y_train)
kNN_score = classifier_kNN.score(X_test, y_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
classifier_RF.fit(X_train, y_train)
RF_score = classifier_RF.score(X_test, y_test)

'''
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((891,1)).astype(int), values = X, axis= 1)
X_opt = X[:, :]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 2, 3 , 4 , 5, 7, 8, 9, 10, 11, 12, 13]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 2, 3 , 4 , 7, 8, 9, 10, 11, 12, 13]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
'''