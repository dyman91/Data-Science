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
dataset = pd.read_csv('train.csv')
y = dataset.iloc[:, 1].values
dataset = dataset.drop(['Survived','PassengerId','Ticket', 'Cabin', 'Embarked'],1)
# Read test Dataset
dataset_test = pd.read_csv('test.csv')
y_test = dataset_test.iloc[:, 0].values
dataset_test = dataset_test.drop(['PassengerId','Ticket', 'Cabin', 'Embarked'],1)


# Names preprocessing
Names = dataset['Name'].str.split(', ').str[1].str.split('.').str[0]
Titles = ['Mr', 'Mrs', 'Miss', 'Master', 'Dr']
Names = Names.apply(lambda i: i if i in Titles else 'None')
dataset['Name'] = Names

Names = dataset_test['Name'].str.split(', ').str[1].str.split('.').str[0]
Names = Names.apply(lambda i: i if i in Titles else 'None')
dataset_test['Name'] = Names

X = dataset.iloc[:,:].values
X_test = dataset_test.iloc[:,:].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, [3]])
X[:, [3]] = imputer.transform(X[:, [3]])

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test[:, [3]])
X_test[:, [3]] = imputer.transform(X_test[:, [3]])

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test[:, [4]])
X_test[:, [4]] = imputer.transform(X_test[:, [4]])

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test[:, [6]])
X_test[:, [6]] = imputer.transform(X_test[:, [6]])



# Encoding categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
labelencoder_X2 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
X_test[:, 1] = labelencoder_X1.transform(X_test[:, 1])
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
X_test[:, 2] = labelencoder_X2.transform(X_test[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1,2])
X = onehotencoder.fit_transform(X).toarray()
X_test = onehotencoder.transform(X_test).toarray()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_train = StandardScaler()
sc_test = StandardScaler()
X = sc_train.fit_transform(X)
X_test = sc_test.fit_transform(X_test)

# Opt features
X = X[:, [0, 1, 2, 3 , 6, 7, 8, 9, 10, 11, 12]]
X_test = X_test[:, [0, 1, 2, 3 , 6, 7, 8, 9, 10, 11, 12]]

# Fitting Random Forest Classification to the Training set
#from sklearn.ensemble import RandomForestClassifier
#classifier_RF = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
#classifier_RF.fit(X, y)
#y_pred = np.array(classifier_RF.predict(X_test))

#ANN
# keras
import keras 
from keras.models import Sequential
from keras.layers import Dense

# initialize ANN
classifier = Sequential()

# Add input and first hidden layer
classifier.add(Dense(output_dim = 4, init = 'uniform', activation='relu', input_dim = 11))

# Add second hidden layer
classifier.add(Dense(output_dim = 4, init = 'uniform', activation='relu'))

# Add output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation='sigmoid'))

# Compiling ANN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])

# Fitting ANN
classifier.fit(X, y, batch_size = 32, epochs = 21)

# Predict results
y_pred = np.array(classifier.predict(X_test))
y_pred = (y_pred > 0.5)
y_pred = y_pred*1
y_pred = y_pred[:,0]

# Write result into array
result = np.append([y_test], [y_pred], axis=0)
result = np.transpose(result)

# Save result to csv
csv_content = pd.DataFrame(data=result[:,:], columns=['PassengerId','Survived'])
csv_content.to_csv('result.csv', index = False)