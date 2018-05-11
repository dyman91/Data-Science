# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 13:28:23 2018

@author: lukasz.dymanski
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read training Dataset
dataset = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
dataset_cat = pd.read_csv('train.csv')
dataset = dataset[dataset.GrLivArea < 4500]
dataset_cat = dataset_cat[dataset_cat.GrLivArea < 4500]
dataset = dataset.drop(['Id','MSZoning', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
       'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinType2','Heating',
       'HeatingQC', 'CentralAir', 'Electrical',
       'KitchenQual',
       'Functional','FireplaceQu', 'GarageType',
       'GarageFinish','GarageQual',
       'GarageCond', 'PavedDrive',
       'PoolQC',
       'Fence', 'MiscFeature','SaleType',
       'SaleCondition', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'OverallCond',
       'KitchenAbvGr', 'BsmtFinSF1', 'BsmtFinSF2', 'MSSubClass', 'LowQualFinSF', 'BsmtHalfBath', 'EnclosedPorch',
       '3SsnPorch','ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold', 'TotRmsAbvGrd', 'YearRemodAdd', 'WoodDeckSF', 'MoSold'], 1)

#dataset = dataset[dataset.GrLivArea < 4500]

dataset_cat = dataset_cat.drop(['Id','LotFrontage', 'LotArea','OverallQual', 'YearBuilt','OverallCond', 'MasVnrArea',
       'KitchenAbvGr', 'BsmtFinSF1', 'BsmtFinSF2', 'MSSubClass', 'LowQualFinSF', 'BsmtHalfBath', 'EnclosedPorch',
       '3SsnPorch','ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold', 'TotRmsAbvGrd', 'YearRemodAdd', 'WoodDeckSF',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
       'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'Fireplaces',
       'GarageYrBlt','GarageCars', 'GarageArea', 'OpenPorchSF','MoSold', 'Alley', 'PoolQC', 'Fence', 'MiscFeature',
       'MSZoning', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
       'MasVnrType', 'ExterCond', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinType2','Heating',
       'CentralAir', 'Electrical',
       'Functional','FireplaceQu', 'GarageType',
       'GarageFinish','GarageQual',
       'GarageCond', 'PavedDrive',
       'PoolQC',
       'Fence', 'MiscFeature','SaleType',
       'SaleCondition', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'OverallCond', 'SalePrice'], 1)

#corrmat = dataset.corr()
#f, ax = plt.subplots(figsize=(12, 9))
#sns.heatmap(corrmat, vmax=.8, square=True);
#Checking for missing data
#NAs = pd.concat([dataset.isnull().sum()], axis=1, keys=['Train'])
#NAs[NAs.sum(axis=1) > 0]
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values
X_cat = dataset_cat.iloc[:,:].values


# Encoding categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
labelencoder_X2 = LabelEncoder()
labelencoder_X3 = LabelEncoder()
labelencoder_X4 = LabelEncoder()
X_cat[:, 0] = labelencoder_X1.fit_transform(X_cat[:, 0])
X_cat[:, 1] = labelencoder_X2.fit_transform(X_cat[:, 1])
X_cat[:, 2] = labelencoder_X3.fit_transform(X_cat[:, 2])
X_cat[:, 3] = labelencoder_X4.fit_transform(X_cat[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [0,1,2,3])
X_cat = onehotencoder.fit_transform(X_cat).toarray()

#Merge arrays
X = np.concatenate((X,X_cat), axis = 1)

#import statsmodels.formula.api as sm
#X = np.append(arr = np.ones((1458,1)).astype(int), values = X, axis= 1)
#X_opt = np.delete(X[:, :], 21, axis=1)
#X_opt = np.delete(X_opt[:, :], 23, axis=1)
#X_opt = np.delete(X_opt[:, :], 34, axis=1)


regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Fitting Random Forest Regression to the dataset
#from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators = 20, random_state = 0)
#regressor.fit(X_train, y_train)

# Fitting XGBoost to the Training set
from xgboost import XGBRegressor
regressor = XGBRegressor(n_estimators = 300, learning_rate = 0.07)
regressor.fit(X_train, y_train)

# Fitting Simple Linear Regression to the Training set
#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(X_train, y_train)
               
# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Evaluate model
from sklearn.metrics import mean_squared_error, r2_score

#rms = [np.sqrt(mean_squared_error(np.log(y_test), np.log(y_pred)))]
# Prints R2 and RMSE scores

print('R2: {}'.format(r2_score(y_pred, y_test)))
print('RMSE log: {}'.format(np.sqrt(mean_squared_error(np.log1p(y_pred), np.log1p(y_test)))))
print('RMSE: {}'.format(np.sqrt(mean_squared_error(y_pred, y_test))))
