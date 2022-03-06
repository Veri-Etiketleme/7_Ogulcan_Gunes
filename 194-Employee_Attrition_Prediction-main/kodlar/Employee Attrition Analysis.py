# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 21:37:41 2021

@author: Singh
"""

#-----------------Employee_Attrition_Prediction--------------

#importing libraries
import pandas as pd
import seaborn as sns

#import data
HR_Data = pd.read_csv('E:/Data Analytics/Project/HR_Data.csv')

#replace 'left' with 'Attrition' 
#HR_Data = HR_Data.rename(columns={'left':'Attrition'})

#Analyze data
sns.countplot(x="Attrition",data=HR_Data)
sns.countplot(x="Attrition",hue ='role',data=HR_Data)
sns.countplot(x="Attrition",hue ='salary',data=HR_Data)

#data type for each column
HR_Data.info()
#Check fro missing values
HR_Data.isnull().any()

#one-hot-encoding, using dummies var to handle categorical var
#pd.get_dummies(HR_Data["salary"])
S_Dummy = pd.get_dummies(HR_Data["salary"],drop_first=True)

#pd.get_dummies(HR_Data['role'])
R_Dummy = pd.get_dummies(HR_Data['role'],drop_first=True)

#concatinate data into dataset
HR_Data = pd.concat([HR_Data,S_Dummy,R_Dummy], axis=1)

HR_Data.drop(['role','salary'],axis=1,inplace=True)

#Spliting data into training and testing
X = HR_Data.drop("left",axis=1)
y = HR_Data["left"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=1/3, random_state = 0)

#Fitting logistic regression to training set
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

#predict test set
y_predict = logmodel.predict(X_test)

#Accuracy of model wrt confusion matrix because dev var has data in binary form
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_predict)

#Accuracy = (3552+466)/(3624+302+680+466) => 79.2%

# coefficient
print(logmodel.coef_)
# intercept
print(logmodel.intercept_)

#--------------To Improve the accuracy of the model use Backward elimination---------------

import statsmodels.api as sm
import numpy as nm 

HR_Data_1 = HR_Data
HR_Data_1.head(5)

x1 = HR_Data_1.drop("left",axis=1)
y1 = HR_Data_1["left"]

x1 = nm.append(arr = nm.ones((14999,1)).astype(int), values=x1, axis=1)

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,12,13,16,17,18]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,12,13,17,18]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,12,13,18]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,12,13]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,13]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

#Building logisticRegression model

from sklearn.model_selection import train_test_split
x_BE_train, x_BE_test, y_BE_train, y_BE_test= train_test_split(x_opt, y1, test_size=0.3, random_state=0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_BE_train, y_BE_train)

y_BE_predict = logmodel.predict(x_BE_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_BE_test,y_BE_predict)

#Accuracy = (3209+365)/(3209+253+673+365) => 79.4%

#Calculating the coefficients:
print(logmodel.coef_)

#Calculating the intercept:
print(logmodel.intercept_)
