# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 14:44:17 2018

@author: lukasz.dymanski
"""

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import WordPunctTokenizer

def basicVisualization(df):
    # Visualization
    order = list(df['language'].unique())
    order.sort()
    
    fig, axs = plt.subplots(nrows = 2, figsize=(20, 10))
    sns.countplot(x="language", data=df, ax = axs[0], order = order)
    sns.barplot(x = df.groupby('language').agg('median').index, y ='body_len', data=df.groupby('language').agg('median'), ax = axs[1])
    axs[0].set_title('Number of samples per language')
    axs[0].set_ylabel('Number of samples')
    axs[1].set_title('Median body lenght per language')
    axs[1].set_ylabel('Median value of body lenght')
    

def clean_dataset(df):
    # Remove comments starting with type of //comment and /* comment */
    for lang in {'JavaScript', 'Java', 'Swift', 'C++', 'Rust', 'C', 'Scala','Go', 'Kotlin', 'PHP', 'Perl'}: 
        df.loc[df['language']==lang,'file_body'] = df[df['language']==lang]['file_body'].str.replace(re.compile("/\*.*?\*/",re.DOTALL ) ,"" ,regex=True)
        df.loc[df['language']==lang,'file_body'] = df[df['language']==lang]['file_body'].str.replace(re.compile("//.*?\n" ) ,"" ,regex=True)
    for lang in {'Python', 'Ruby', 'R', 'Julia'}:
        df.loc[df['language']==lang,'file_body'] = df[df['language']==lang]['file_body'].str.replace(re.compile("#.*?\n" ) ,"" ,regex=True)
    for lang in {'MATLAB'}:
        df.loc[df['language']==lang,'file_body'] = df[df['language']==lang]['file_body'].str.replace(re.compile("%.*?\n" ) ,"" ,regex=True)
    for lang in {'Haskell'}:
        df.loc[df['language']==lang,'file_body'] = df[df['language']==lang]['file_body'].str.replace(re.compile("--.*?\n" ) ,"" ,regex=True)
    
    # Remove digits
    df['file_body'] = df['file_body'].str.replace(re.compile('\d+(.\d+)?'), "CYFRA", regex=True)
   
    # Replace content inside quotes with hardcoded string
    df['file_body'] = df['file_body'].str.replace(re.compile('".*?"') ,"SLOWO" ,regex=True)
    df['file_body'] = df['file_body'].str.replace(re.compile("'.*?'") ,"SLOWO" ,regex=True)
    
    # Remove tabs and new lines
    df['file_body'] = df['file_body'].str.replace(re.compile('\n|\t') ," " ,regex=True)
    
    return df


def build_input_model():
    # Read dataset
    df = pd.read_csv('data.csv')
    df['body_len'] = df['file_body'].apply(str).apply(len)
    
    # Data Engineering
    df = clean_dataset(df)
    
    return df

def my_tokenizer(doc):
    # Use WordPunctTokenizer to split text into tokens
    my_tokenizer = WordPunctTokenizer()
    
    return my_tokenizer.tokenize(doc)
    
    
def create_bag_of_words_model(df):
   #Create bag of words model 
   vectorizer = CountVectorizer(tokenizer = my_tokenizer, max_features = 1500)
   bag_of_words = vectorizer.fit_transform(df['file_body'].values.astype('U'))
   tfidf_transformer = TfidfTransformer(use_idf=False).fit(bag_of_words)
   bag_of_words_norm = tfidf_transformer.transform(bag_of_words).toarray()
   vocab = vectorizer.get_feature_names() 
   
   return vocab, bag_of_words_norm

def train_model(X_train, y_train, X_test, y_test):
    # Train model with Random Forest Classifier
    classifier = RandomForestClassifier(n_estimators = 200)
    classifier.fit(X_train, y_train)

    #Predict values for test set
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return y_pred, accuracy

if __name__ == '__main__':
    # Build model input
    df = build_input_model()
	
    #Visualisation
    basicVisualization(df)
	
    # Encode languages values in dataset and shuffle data
    lb = LabelEncoder()
    df['language'] = lb.fit_transform(df['language'].values)
    df = df.sample(frac=1)
	
    # Build bag of words model 
    vocab, bag_of_words = create_bag_of_words_model(df)
	
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(bag_of_words, df['language'].values, test_size = 0.20, random_state = 0)
    
	# Train model
    y_pred, accuracy = train_model(X_train,y_train, X_test, y_test)
	
    # Presenation of results
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
    results = pd.DataFrame( data={'precision': precision, 'recall': recall, 'fscore': fscore})
    results.index = lb.inverse_transform(results.index)
    results.plot( kind="bar", figsize = (20, 5), title = 'Results per language, overall accuracy: '+"{:.2%}".format(accuracy)).legend(bbox_to_anchor=(1, 1))