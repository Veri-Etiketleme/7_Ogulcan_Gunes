# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 18:53:26 2020

@author: Ash
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

def scale(partitions, scaler='MinMaxScaler', df=pd.DataFrame(), to_float=False, return_df=False):
  #return_df == True, allora output = scaler, df
  #return_df == False, allora output = df
  from sklearn.preprocessing import RobustScaler
  from sklearn.preprocessing import MinMaxScaler
  from sklearn.preprocessing import StandardScaler
  
  if scaler == 'RobustScaler':
    f_transformer = RobustScaler()
  elif scaler == 'MinMaxScaler':
    f_transformer = MinMaxScaler(feature_range=(0, 1))
  elif scaler == 'StandardScaler':
    f_transformer = StandardScaler()
  
  #partitions = 'all_df', le fa tutte insieme e trasforma il df in un numpy.array
  if partitions == 'all_df':
    if to_float == True:
      df = df.astype('float32')
    if df.empty == True:
      X = df.copy()
    #tutto df deve essere con float32
    df_col = df.columns
    df = f_transformer.fit_transform(df.values) #ne esce un inspiegabile numpy array
    df = pd.DataFrame(df)
    df.columns = df_col
    if return_df == True:
      return f_transformer, df
    else:
      X = df.copy()
    return f_transformer
  else:
    #partitions = ['col1', 'col2'], fa solo partizioni specificate
    pass

trXnew = []

# Importing the dataset
dataset = pd.read_csv('Train.csv')
dataset1 = pd.read_csv('Train.csv')

X5 = dataset.iloc[:, [2,7,8,14,15,16,19,21,22]].values
y = dataset.iloc[:, 39].values

col_mean = np.nanmean(X5, axis = 0)
inds = np.where(np.isnan(X5)) 
X5[inds] = np.take(col_mean, inds[1])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X5, y, test_size = 0.25, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sc = StandardScaler()
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)   

constant_value = len(dataset.columns)

from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(loss='ls', learning_rate=0.001, n_estimators=55,
                                subsample=1.0, criterion='friedman_mse',
                                min_samples_split=100, min_samples_leaf=10, 
                                min_weight_fraction_leaf=0, max_depth=8, 
                                min_impurity_decrease=0.001, min_impurity_split=None, 
                                init=None, random_state=43, max_features="auto", 
                                alpha=0.1, verbose=0, max_leaf_nodes=None, 
                                warm_start=False, 
                                validation_fraction=0.01, n_iter_no_change=55, tol=1e-4)

reg.fit(X_train, y_train)
preds = reg.predict(X_test)

ac1 = reg.score(X_test,y_test)
rms = sqrt(mean_squared_error(y_test, preds))
print('root',rms)
print('Accuracy is: ',ac1)
score= 100 * max(0, 1 - rms)
print('score1=',score)
