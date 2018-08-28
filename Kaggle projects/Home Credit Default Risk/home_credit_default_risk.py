# -*- coding: utf-8 -*-
"""
Created on Wed May 30 10:14:48 2018

@author: lukasz.dymanski
"""
import numpy as np
import pandas as pd 
from sklearn import preprocessing
#from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
#import seaborn as sns

# read datasets
training_set = pd.read_csv('application_train.csv')
test_set = pd.read_csv('application_test.csv')
training_set = training_set.set_index('SK_ID_CURR')
test_set = test_set.set_index('SK_ID_CURR')
columns = training_set.columns
bureau = pd.read_csv('bureau.csv')
bureau = bureau.set_index('SK_ID_BUREAU')
bureau_balance = pd.read_csv('bureau_balance.csv')
previous_app = pd.read_csv('previous_application.csv')
previous_app = previous_app.drop(['SK_ID_PREV'], axis=1)
pos_cash = pd.read_csv('POS_CASH_balance.csv').drop(['SK_ID_PREV'], axis=1)
installments_payments = pd.read_csv('installments_payments.csv').drop(['SK_ID_PREV'], axis=1)
credit_card = pd.read_csv('credit_card_balance.csv').drop(['SK_ID_PREV'], axis=1)

# Columns extraction
y_train = training_set.TARGET
training_set = training_set.drop(['TARGET'], axis=1)
#test_set = test_set[columns.App_Columns.values]

def encodeCategoricalFeatures(ds):
    categorical_features = [i for i in ds.columns if ds[i].dtype == 'object']
    
    # Encode categorical values
    for col in categorical_features:
        lb = preprocessing.LabelEncoder()
        lb.fit(list(ds[col].values.astype('str')))
        ds[col] = lb.transform(list(ds[col].values.astype('str')))

def installments_payments_fe(installments_payments):
    installments_payments['PAST_DUE_DATE'] = installments_payments['DAYS_ENTRY_PAYMENT'] - installments_payments['DAYS_INSTALMENT']
    installments_payments['BEFORE_DUE_DATE'] = installments_payments['DAYS_INSTALMENT'] - installments_payments['DAYS_ENTRY_PAYMENT']
    installments_payments['UNDERPAID'] = installments_payments['AMT_INSTALMENT'] - installments_payments['AMT_PAYMENT']
    installments_payments['OVERPAID'] = installments_payments['AMT_PAYMENT'] - installments_payments['AMT_INSTALMENT']

def previous_app_fe(previous_app):
    # Days 365.243 values -> nan
    previous_app['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    previous_app['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    previous_app['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    previous_app['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    previous_app['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    previous_app['APP_CREDIT_PERC'] = previous_app['AMT_APPLICATION'] / previous_app['AMT_CREDIT']
    previous_app['APP_ANNUITY_TO_CREDIT_RATIO'] = previous_app['AMT_ANNUITY'] / previous_app['AMT_CREDIT']
    previous_app['APP_CREDIT_TO_GOODS_RATIO'] = previous_app['AMT_CREDIT'] / previous_app['AMT_GOODS_PRICE']
    previous_app['APP_CREDIT_TO_TERM_RATIO'] = previous_app['AMT_CREDIT'] / previous_app['CNT_PAYMENT']
    
def bureau_fe(bureau):
    bureau['NEW_ANNUITY_TO_CREDIT_SUM_RATIO'] = bureau['AMT_ANNUITY'] / bureau['AMT_CREDIT_SUM']
    active = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
    closed = bureau[bureau['CREDIT_ACTIVE'] == 'Closed']
    closed['BUREAU_ENDDATE_INTERVAL'] = closed['DAYS_ENDDATE_FACT'] - closed['DAYS_CREDIT_ENDDATE']  
    active.columns = [str(col) + '_ACTIVE' for col in active.columns]
    closed.columns = [str(col) + '_CLOSED' for col in closed.columns]
    bureau = bureau.join(active)
    bureau = bureau.join(closed)
    return bureau

def pos_cash_fe(pos_cash):
    pos_cash['POS_CASH_CNT_INSTALMENT_INTERVAL'] = pos_cash['CNT_INSTALMENT'] / pos_cash['CNT_INSTALMENT_FUTURE']
    pos_cash['POS_CASH_PAID_LATE'] = (pos_cash['SK_DPD'] > 0).astype(int)
    pos_cash['POS_CASH_PAID_LATE_WITH_TOLERANCE'] = (pos_cash['SK_DPD_DEF'] > 0).astype(int)

def credit_card_fe(credit_card):
    credit_card['CC_ATM_TO_TOTAL'] = credit_card['AMT_DRAWINGS_ATM_CURRENT'] / credit_card['AMT_DRAWINGS_CURRENT']
    credit_card['CC_DRAWINGS_TO_PAY'] = credit_card['AMT_DRAWINGS_CURRENT'] / credit_card['AMT_PAYMENT_CURRENT']
    
    
    
# Encode categorical values in training set and test set
categorical_features = [i for i in training_set.columns if training_set[i].dtype == 'object']

for col in categorical_features:
    lb = preprocessing.LabelEncoder()
    lb.fit(list(training_set[col].values.astype('str')) + list(test_set[col].values.astype('str')))
    training_set[col] = lb.transform(list(training_set[col].values.astype('str')))
    test_set[col] = lb.transform(list(test_set[col].values.astype('str')))

#Feature engineering
installments_payments_fe(installments_payments)
previous_app_fe(previous_app)
bureau = bureau_fe(bureau)
pos_cash_fe(pos_cash)
credit_card_fe(credit_card)
   
#Encode categorical values in additional datasets
encodeCategoricalFeatures(bureau)
encodeCategoricalFeatures(bureau_balance)
encodeCategoricalFeatures(previous_app)
encodeCategoricalFeatures(pos_cash) 
encodeCategoricalFeatures(installments_payments)
encodeCategoricalFeatures(credit_card)          



#Extract column names
installments_payments_columns = installments_payments.columns.drop('SK_ID_CURR')
bureau_columns = bureau.columns.drop('SK_ID_CURR')
bureau_balance_columns = bureau_balance.columns.drop('SK_ID_BUREAU')
previous_app_columns = previous_app.columns.drop('SK_ID_CURR')
credit_card_columns = credit_card.columns.drop('SK_ID_CURR')
pos_cash_columns = pos_cash.columns.drop('SK_ID_CURR')

#Merge datasets
def joinByMean(feature, main_set, df, df_name, key='SK_ID_CURR'):
    df_temp = df[[key, feature]]
    df_agg = df_temp.groupby(key).agg({feature: [sum, 'mean', min, max]})
    df_agg.columns = [df_name.join(x) for x in df_agg.columns.ravel()]
    return main_set.join(df_agg)

def joinSize(main_set, df, df_name, key='SK_ID_CURR'):
    df_temp = df[[key]]
    df_agg = df_temp.groupby(key).agg({key: ['size']})
    df_agg.columns = [df_name.join(x) for x in df_agg.columns.ravel()]
    return main_set.join(df_agg)

training_set = joinSize(training_set, bureau, 'bureau')
training_set = joinSize(training_set, previous_app, 'previous_app')
training_set = joinSize(training_set, pos_cash, 'pos_cash')
training_set = joinSize(training_set, installments_payments, 'installments_payments')
training_set = joinSize(training_set, credit_card, 'credit_card')
test_set = joinSize(test_set, bureau, 'bureau')
test_set = joinSize(test_set, previous_app, 'previous_app')
test_set = joinSize(test_set, pos_cash, 'pos_cash')
test_set = joinSize(test_set, installments_payments, 'installments_payments')
test_set = joinSize(test_set, credit_card, 'credit_card')

for col in bureau_balance_columns:
    bureau = joinByMean(col, bureau, bureau_balance, 'bureau_balance', key = 'SK_ID_BUREAU')

for col in bureau_columns:
    training_set = joinByMean(col, training_set, bureau, 'bureau')
    test_set = joinByMean(col, test_set, bureau, 'bureau')
   
for col in previous_app_columns:
    training_set = joinByMean(col, training_set, previous_app, 'previous_app')
    test_set = joinByMean(col, test_set, previous_app, 'previous_app')
    
for col in pos_cash_columns:
    training_set = joinByMean(col, training_set, pos_cash, 'pos_cash')
    test_set = joinByMean(col, test_set, pos_cash, 'pos_cash')

for col in installments_payments_columns:
    training_set = joinByMean(col, training_set, installments_payments, 'installments_payments')
    test_set = joinByMean(col, test_set, installments_payments, 'installments_payments')
    
for col in credit_card_columns:
    training_set = joinByMean(col, training_set, credit_card, 'credit_card')
    test_set = joinByMean(col, test_set, credit_card, 'credit_card')

def addNewFeatures(df):
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['NEW_ANNUITY_TO_CREDIT_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_EXT_SOURCES_SUM'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis=1)
    df['NEW_EXT_SOURCES_MIN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
    df['NEW_EXT_SOURCES_MAX'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
    df['NEW_INCOME_TO_ANNUITY_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_ANNUITY']
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_EMPLOY_TO_BIRTH_INTERVEL'] = df['DAYS_EMPLOYED'] - df['DAYS_BIRTH']
    df['NEW_REGISTRATION_TO_BIRTH_INTERVEL'] = df['DAYS_REGISTRATION'] - df['DAYS_BIRTH']
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['NEW_SHORT_EMPLOYMENT'] = (df['DAYS_EMPLOYED'] < -2000).astype(int)
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    df['NEW_DOC_IND_AVG'] = df[docs].mean(axis=1)
    df['NEW_DOC_IND_SUM'] = df[docs].sum(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_LIVE_IND_STD'] = df[live].std(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    return df
    
training_set = addNewFeatures(training_set)
test_set = addNewFeatures(test_set)

del(bureau)
del(bureau_balance)
del(credit_card)
del(installments_payments)
del(pos_cash)
del(previous_app)


def kfoldLightGBMTraining(folds, training_set, test_set, y_train):
    preds = np.zeros(test_set.shape[0])
    for fold, (train_idx, valid_idx) in enumerate(folds.split(training_set, y_train)):
        kfold_train_set = training_set.iloc[train_idx]
        kfold_valid_set = training_set.iloc[valid_idx]
        kfold_train_y = y_train.iloc[train_idx]
        kfold_valid_y = y_train.iloc[valid_idx]
        
        classifier = LGBMClassifier(
                nthread=2,
                n_estimators=10000,
                learning_rate=0.02,
                num_leaves=44,
                feature_fraction = 0.1,
                bagging_fraction = 0.91,
                colsample_bytree=0.9497036,
                subsample=0.8715623,
                max_depth=8,
                reg_alpha=0.081545473,
                reg_lambda=2.735294,
                min_split_gain=0.092,
                min_child_weight=49.3259775,
                silent=-1,
                verbose=-1,
                n_interations = 1400)
        classifier.fit(
            kfold_train_set, kfold_train_y,
            eval_set=[(kfold_train_set, kfold_train_y), (kfold_valid_set, kfold_valid_y)],
            eval_metric= 'auc',
            verbose= 100,
            early_stopping_rounds= 100)
        
        #Predict results
        y_pred = classifier.predict_proba(test_set)
        y_pred = y_pred[:,1]
        preds = np.vstack((preds, y_pred))
    return preds

#Predict results
folds = KFold(n_splits= 5, shuffle=True, random_state=0)
preds = kfoldLightGBMTraining(folds, training_set, test_set, y_train)
preds = np.sum(preds,axis=0)/5

# Save results into csv
x_test = pd.read_csv('application_test.csv')['SK_ID_CURR']
csv_content = pd.DataFrame({'SK_ID_CURR': x_test, 'TARGET': preds})
csv_content.to_csv('result.csv', index = False)