from sklearn.utils import shuffle
import tensorflow as tf 
import pandas as pd 
import numpy as np 

input_data = pd.read_csv('data/creditcard.csv')

input_data.loc[input_data.Class == 0, 'Normal'] = 1
input_data.loc[input_data.Class == 1, 'Normal'] = 0
input_data = input_data.rename(columns={'Class': 'Fraud'})

Fraud = input_data[input_data.Fraud == 1]
Normal = input_data[input_data.Normal == 1]

X_train = Fraud.sample(frac=0.8)
X_train = pd.concat([X_train, Normal.sample(frac = 0.8)], axis = 0)
X_test = input_data.loc[~input_data.index.isin(X_train.index)]