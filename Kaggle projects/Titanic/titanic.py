# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:05:53 2018

@author: lukasz.dymanski
"""

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

def load_data():
    # laod training and test sets
    training_set = pd.read_csv('train.csv').set_index('PassengerId')
    test_set = pd.read_csv('test.csv').set_index('PassengerId')
    y_train = training_set.Survived
    
    training_set = training_set.drop(['Ticket', 'Cabin','Survived'],1)
    test_set = test_set.drop(['Ticket', 'Cabin'],1)
    #training_set = training_set.sample(frac=1)
    training_set, test_set, X_train, X_test = clean_data(training_set, test_set)
    
    return training_set, test_set, y_train, X_train, X_test

def clean_data(training_set, test_set):
    
    # Extract title from name
    for df in (training_set, test_set):
        Names = df['Name'].str.split(', ').str[1].str.split('.').str[0]
        Titles = ['Mr', 'Mrs', 'Miss', 'Master', 'Dr']
        Names = Names.apply(lambda i: i if i in Titles else 'None')
        df['Name'] = Names
    
    # Encode categorical values
    for col in ['Name', 'Sex', 'Embarked' ]:
        lb = LabelEncoder()
        lb.fit(list(training_set[col].values.astype('str')) + list(test_set[col].values.astype('str')))
        training_set[col] = lb.transform(list(training_set[col].values.astype('str')))
        test_set[col] = lb.transform(list(test_set[col].values.astype('str')))

    # One hot encoding
    training_set = pd.get_dummies(training_set, columns=['Name', 'Sex', 'Embarked' ])
    test_set = pd.get_dummies(test_set, columns=['Name', 'Sex', 'Embarked' ])
    training_set, test_set = training_set.align(test_set, join='left', axis=1)
    
    # Deal with missing data
    training_set = training_set.fillna(training_set.mean())
    test_set = test_set.fillna(test_set.mean())
    test_set = test_set.fillna(0)
    
    # Normalize data
    sc_train = StandardScaler()
    sc_test = StandardScaler()
    X_train= sc_train.fit_transform(training_set)
    X_test = sc_test.fit_transform(test_set)
    
    return training_set, test_set, X_train, X_test    
    
def kfoldXGboostTraining(folds, training_set, test_set, y_train):
    
    # Train XGBoost model with kFold technique
    preds = np.zeros(test_set.shape[0])
    for fold, (train_idx, valid_idx) in enumerate(folds.split(training_set, y_train)):
        kfold_train_set = training_set.iloc[train_idx]
        kfold_valid_set = training_set.iloc[valid_idx]
        kfold_train_y = y_train.iloc[train_idx]
        kfold_valid_y = y_train.iloc[valid_idx]
        
        classifier = XGBClassifier()
        classifier.fit(kfold_train_set, kfold_train_y,
            eval_set=[(kfold_train_set, kfold_train_y), (kfold_valid_set, kfold_valid_y)],
            eval_metric= 'auc',
            verbose= 100,
            early_stopping_rounds= 5)
        
        #Predict results
        y_pred = classifier.predict(test_set)
        preds = np.vstack((preds, y_pred))
    
    
    preds = np.round(np.mean(preds[1:],axis=0)).astype(int)
    
    return preds

def annTraining(X_train, X_test, y_train):
    
    # initialize ANN
    classifier = Sequential()

    # Add input and first hidden layer
    classifier.add(Dense(output_dim = 4, init = 'uniform', activation='relu', input_dim = 17))

    # Add second hidden layer
    classifier.add(Dense(output_dim = 4, init = 'uniform', activation='relu'))

    # Add output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation='sigmoid'))
    
    # Compiling ANN
    classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])

    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20, verbose=1, mode='auto')
    callbacks_list = [earlystop]
    
    # Fitting ANN
    classifier.fit(X_train, y_train.iloc[:], batch_size = 32, nb_epoch=50, callbacks=callbacks_list, validation_split=0.2 )
    
    
    # Predict results
    y_pred = np.array(classifier.predict(X_test))
    y_pred = (y_pred > 0.5)
    y_pred = y_pred*1
    y_pred = y_pred[:,0]
    
    return y_pred

if __name__ == '__main__':

    # Prepare data     
    training_set, test_set, y_train, X_train, X_test = load_data()
    
    # Train model
    folds = KFold(n_splits= 5, shuffle=True, random_state=1001)
    #y_pred_xgb = kfoldXGboostTraining(folds, training_set, test_set, y_train)
    y_pred_ann = annTraining(X_train, X_test, y_train)
    
    # Save result to csv
    x_test = pd.read_csv('test.csv')['PassengerId']
    csv_content = pd.DataFrame({'PassengerId': x_test, 'Survived': y_pred_ann})
    csv_content.to_csv('result.csv', index = False)
    
    
    