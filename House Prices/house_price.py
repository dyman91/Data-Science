# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 14:47:22 2018

@author: lukasz.dymanski
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read training Dataset
dataset_train = pd.read_csv('train.csv')
dataset_train_cat = pd.read_csv('train.csv')
dataset_train = dataset_train[dataset_train.GrLivArea < 4500]
dataset_train_cat = dataset_train_cat[dataset_train_cat.GrLivArea < 4500]
dataset_train = dataset_train.drop(['Id','MSZoning', 'Street',
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
       '3SsnPorch','ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold', 'TotRmsAbvGrd', 'YearRemodAdd', 'WoodDeckSF'], 1)
#dataset_train = dataset_train[dataset_train.GrLivArea < 4500]
dataset_train_cat = dataset_train_cat.drop(['Id','LotFrontage', 'LotArea','OverallQual', 'YearBuilt','OverallCond', 'MasVnrArea',
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
       'HeatingQC', 'CentralAir', 'Electrical',
       'Functional','FireplaceQu', 'GarageType',
       'GarageFinish','GarageQual',
       'GarageCond', 'PavedDrive',
       'PoolQC',
       'Fence', 'MiscFeature','SaleType',
       'SaleCondition', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'OverallCond', 'SalePrice'], 1)
X_train = dataset_train.iloc[:, :-1].values
y_train = dataset_train.iloc[:, -1].values
X_train_cat = dataset_train_cat.iloc[:, :].values

# Read test Dataset
dataset_test = pd.read_csv('test.csv')
dataset_test_cat = pd.read_csv('test.csv')
y_test = dataset_test.iloc[:, 0].values
dataset_test = dataset_test.drop(['Id','MSZoning', 'Street',
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
       '3SsnPorch','ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold', 'TotRmsAbvGrd', 'YearRemodAdd', 'WoodDeckSF'], 1)
dataset_test_cat = dataset_test_cat.drop(['Id','LotFrontage', 'LotArea','OverallQual', 'YearBuilt','OverallCond', 'MasVnrArea',
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
       'HeatingQC', 'CentralAir', 'Electrical',
       'Functional','FireplaceQu', 'GarageType',
       'GarageFinish','GarageQual',
       'GarageCond', 'PavedDrive',
       'PoolQC',
       'Fence', 'MiscFeature','SaleType',
       'SaleCondition', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'OverallCond'], 1)
X_test = dataset_test.iloc[:,:].values

dataset_test_cat = dataset_test_cat.fillna('TA')

X_test_cat = dataset_test_cat.iloc[:,:].values
#Checking for missing data
NAs = pd.concat([dataset_test.isnull().sum(), dataset_train.isnull().sum()], axis=1, keys=[ 'Test', 'Train'])
NAs[NAs.sum(axis=1) > 0]

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test[:, :])
X_test[:, :] = imputer.transform(X_test[:, :])

# Encoding categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
labelencoder_X2 = LabelEncoder()
labelencoder_X3 = LabelEncoder()
labelencoder_X4 = LabelEncoder()
X_train_cat[:, 0] = labelencoder_X1.fit_transform(X_train_cat[:, 0])
X_test_cat[:, 0] = labelencoder_X1.transform(X_test_cat[:, 0])
X_train_cat[:, 1] = labelencoder_X2.fit_transform(X_train_cat[:, 1])
X_test_cat[:, 1] = labelencoder_X2.transform(X_test_cat[:, 1])
X_train_cat[:, 2] = labelencoder_X3.fit_transform(X_train_cat[:, 2])
X_test_cat[:, 2] = labelencoder_X3.transform(X_test_cat[:, 2])
X_train_cat[:, 3] = labelencoder_X4.fit_transform(X_train_cat[:, 3])
X_test_cat[:, 3] = labelencoder_X4.transform(X_test_cat[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [0,1,2,3])
X_train_cat = onehotencoder.fit_transform(X_train_cat).toarray()
X_test_cat = onehotencoder.transform(X_test_cat).toarray()


#Merge 
X_train = np.concatenate((X_train,X_train_cat), axis = 1)
X_test = np.concatenate((X_test,X_test_cat), axis = 1)

# Fitting Simple Linear Regression to the Training set
#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(X_train, y_train)

# Fitting XGBoost to the Training set
from xgboost import XGBRegressor
regressor = XGBRegressor(n_estimators = 300, learning_rate = 0.07)
regressor.fit(X_train, y_train)

# Fitting Random Forest Regression to the dataset
#from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators = 20, random_state = 0)
#regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Save result to csv

#csv_content = pd.DataFrame(data = [y_test, y_pred], columns=['Id', 'SalePrice'])
csv_content = pd.DataFrame({'Id': y_test, 'SalePrice': y_pred})
csv_content.to_csv('result.csv', index = False)

