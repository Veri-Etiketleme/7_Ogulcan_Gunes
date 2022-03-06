#!/usr/bin/env python
# coding: utf-8

# In[46]:


import os
import seaborn as sns
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures


# In[8]:


train_data = pd.read_csv('D:\hackearth\employees Dataset\Train.csv')
test_data = pd.read_csv('D:\hackearth\employees Dataset\Test.csv')

print(train_data.head(5))
print(test_data.head(2))


# In[10]:


train_data.shape
#test_data.shape


# In[11]:


test_data.shape


# In[13]:


train_data.isnull().sum()


# In[15]:


train_data.columns


# In[16]:


train_data = train_data.drop(columns=['Employee_ID','VAR1', 'VAR2', 'VAR3', 'VAR4', 'VAR5', 'VAR6',
       'VAR7'])


# In[19]:


train_data.head(5)


# In[20]:


train_data.isnull().sum()


# In[21]:


train_data.shape


# In[29]:


plt.figure(figsize=(12,8))
train_data['Attrition_rate'].hist(bins=15)


# In[33]:


train_data.columns


# In[34]:


#creating dummies values

new_train_data=pd.get_dummies(train_data, columns=columns,drop_first=True)
new_train_data.head()


# In[37]:


X=new_train_data.drop(columns=['Attrition_rate'])
y=new_train_data['Attrition_rate']


# In[39]:


from sklearn.impute import KNNImputer


knn=KNNImputer()
X=knn.fit_transform(new_train_data.drop(columns=['Attrition_rate']))
X

print(X.shape)
print(y.shape)


# In[42]:


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=0)


# In[43]:


scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# In[47]:


lasso=Lasso()
lasso.fit(X_train,y_train)
lasso_pred=lasso.predict(X_test)

Rf=RandomForestRegressor()
Rf.fit(X_train,y_train)
Rf_pred=Rf.predict(X_test)

Lr=LinearRegression()
Lr.fit(X_train,y_train)
Lr_pred=Lr.predict(X_test)

xg=XGBRegressor()
xg.fit(X_train,y_train)
xg_pred=xg.predict(X_test)

r2_score(y_test,xg_pred)

MSE=np.sqrt(mean_squared_error(y_test,Lr_pred))
MSE


# In[48]:


test_data


# In[61]:


new_test_data=test_data.drop(columns=['Employee_ID','VAR1','VAR2','VAR3','VAR4','VAR5','VAR6','VAR7'])
new_test_data


# In[60]:


columns=['Gender','Relationship_Status','Hometown','Unit','Decision_skill_possess','Compensation_and_Benefits']

columns


# In[51]:


new_test_data=pd.get_dummies(new_test_data, columns=columns,drop_first=True)
new_test_data.head()


# In[52]:


new_test_data1=knn.fit_transform(new_test_data)
new_test_data1


# In[53]:


new_test_data1=scaler.fit_transform(new_test_data1)


# In[54]:


print(X_test.shape)
print(new_test_data1.shape)


# In[55]:


predictions=Lr.predict(new_test_data1)
predictions


# In[58]:


predictions


# In[64]:


submission.to_csv(r'D:\hackearth\EmployeeAttrition2.csv', index=False)
print('Submission CSV is ready!')


# In[ ]:




