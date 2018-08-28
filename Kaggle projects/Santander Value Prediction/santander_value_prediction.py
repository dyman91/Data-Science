# -*- coding: utf-8 -*-
"""
Created on Wed May 30 10:14:48 2018

@author: lukasz.dymanski
"""
import numpy as np
import pandas as pd 
from sklearn import preprocessing
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt


def load_input_model():
    # read dataset
    training_set = pd.read_csv('train.csv')
    training_set = training_set.set_index('ID')
    test_set = pd.read_csv('test.csv')
    test_set = test_set.set_index('ID')

    # target extraction
    y_train = training_set.target
    y_train = np.log1p(y_train)
    training_set = training_set.drop(['target'], axis=1)
    
    # data engineering
    training_set, test_set = clean_data(training_set, test_set)

    training_set, test_set = add_new_features(training_set, test_set)
    
    return training_set, test_set, y_train

def clean_data(training_set, test_set):
    
    # remove constant columns
    colsToRemove = []
    for col in training_set.columns:
    	if training_set[col].std() == 0:
    		colsToRemove.append(col)
    		
    training_set.drop(colsToRemove, axis=1, inplace=True)
    test_set.drop(colsToRemove, axis=1, inplace=True)
    
    colsToRemove = [col for col in training_set.columns[2:] if [i[1] for i in list(training_set[col].value_counts().items()) if i[0] == 0][0] >= 4459 * 0.98]
    training_set.drop(colsToRemove, axis=1, inplace=True)
    test_set.drop(colsToRemove, axis=1, inplace=True)
    
    return training_set, test_set

def add_new_features(training_set, test_set):
	# reaplce 0 with NaN
    training_set.replace(0, np.nan, inplace=True)
    test_set.replace(0, np.nan, inplace=True)
    
    features = [f for f in training_set.columns]
    
	# add some statistical features
    for df in [training_set, test_set]:
        df['the_median'] = df[features].median(axis=1)
        df['the_max'] = df[features].max(axis=1)
        df['the_min'] = df[features].min(axis=1)
        df['the_mean'] = df[features].mean(axis=1)
        df['the_sum'] = df[features].sum(axis=1)
        df['the_std'] = df[features].std(axis=1)
        df['the_kur'] = df[features].kurtosis(axis=1)
        df['the_var'] = df[features].var(axis=1)
        df['the_skew'] = df[features].skew(axis=1)
        df['the_nulls'] = df[features].isnull().sum(axis=1)
        df['the_ratio'] = df['the_std'] / df['the_mean']
		
    return training_set, test_set

def log_feature(training_set, test_set, feature):
    
    training_set[feature] = np.log1p(training_set[feature])
    test_set[feature] = np.log1p(test_set[feature])
    
    return training_set, test_set
    
def plot_distribution(trainin_set, feature,color='red'):
    fig, ax  = plt.subplots()
    sns.distplot(trainin_set[feature].dropna(), kde = True, bins=100, color="r", ax = ax)
    plt.show()

def kfoldxgbPrediction(training_set, test_set, y_train, n_folds = 5):
	# train XGBoost machine with k-fold technique
    sub_preds = np.zeros(test_set.shape[0])
    folds = KFold(n_splits= 5, shuffle=True, random_state=1001)
    
    for fold, (train_idx, valid_idx) in enumerate(folds.split(training_set, y_train)):
        kfold_train_set = training_set.iloc[train_idx]
        kfold_valid_set = training_set.iloc[valid_idx]
        kfold_train_y = y_train.iloc[train_idx]
        kfold_valid_y = y_train.iloc[valid_idx]
    
        regressor = XGBRegressor(
                objective='reg:linear',  
                booster='gbtree',
                n_estimators=4000,
                colsample_bylevel=0.5,
                colsample_bytree=0.054, 
                gamma=1.45, 
                learning_rate=0.005, 
                max_delta_step=0,
                max_depth=22, 
                min_child_weight=57, 
                missing=None, 
                n_jobs=1, 
                random_state=455,
                subsample=0.67
                )
        regressor.fit(kfold_train_set, kfold_train_y,
                eval_set=[(kfold_train_set, kfold_train_y), (kfold_valid_set, kfold_valid_y)],
                eval_metric= 'rmse',
                verbose= 100,
                early_stopping_rounds= 100)
        
        #Predict results
        y_pred = regressor.predict(test_set)
        sub_preds = np.vstack((sub_preds, y_pred))
    return sub_preds

def kfoldlgbmPrediction(training_set, test_set, y_train, n_folds = 5):
	# train LGBoost Machine with k-fold technique
    sub_preds = np.zeros(test_set.shape[0])
    folds = KFold(n_splits= 5, shuffle=True, random_state=1001)
    
    for fold, (train_idx, valid_idx) in enumerate(folds.split(training_set, y_train)):
        kfold_train_set = training_set.iloc[train_idx]
        kfold_valid_set = training_set.iloc[valid_idx]
        kfold_train_y = y_train.iloc[train_idx]
        kfold_valid_y = y_train.iloc[valid_idx]
    
        regressor = LGBMRegressor(
                nthread=2,
                n_estimators=4000,
                learning_rate=0.005,
                num_leaves=144,
                feature_fraction = 0.7,
                bagging_fraction = 0.7,
                bagging_frequency = 5,
                bagging_seed = 2018,
                verbose=-1)
        regressor.fit(kfold_train_set, kfold_train_y,
                eval_set=[(kfold_train_set, kfold_train_y), (kfold_valid_set, kfold_valid_y)],
                eval_metric= 'rmse',
                verbose= 100,
                early_stopping_rounds= 100)
        
        #Predict results
        y_pred = regressor.predict(test_set)
        sub_preds = np.vstack((sub_preds, y_pred))
    return sub_preds

def savePredictions(preds, training_set, y_train):
    
    pred = np.sum(preds,axis=0)/10
    pred = np.expm1(pred)
    
    # Write predictions to file 
    x_test = pd.read_csv('test.csv')['ID']
    csv_content = pd.DataFrame({'ID': x_test, 'target': pred})
    csv_content.to_csv('result.csv', index = False)
    
def rmsle(y_pred, y_test): 
    return np.sqrt(np.square(np.log(y_pred + 1) - np.log(y_test + 1)).mean())

if __name__ == '__main__':
    # Load data
    training_set, test_set, y_train = load_input_model()

    # Train model
    preds_xgb = kfoldxgbPrediction(training_set, test_set, y_train)
    preds_lgbm = kfoldlgbmPrediction(training_set, test_set, y_train)
    
    preds =  np.concatenate((preds_xgb, preds_lgbm), axis=0)
    # Write predictions to file
    savePredictions(preds, training_set, y_train)
