#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data=pd.read_excel(r"E:\data.xlsx")
data.shape


# In[4]:


#Checking missing values
data.isnull().sum()


# In[5]:


data.info()


# In[6]:


data.head()


# In[7]:


#Dropping irrelevant columns that are constant throughout
data1=data.drop(["EmployeeCount","EmployeeNumber","Over18","StandardHours"],axis=1)


# In[8]:


data1.shape


# In[9]:


from sklearn.preprocessing import LabelEncoder


# In[10]:


le=LabelEncoder()


# In[11]:


for x in data1:
    if(data1[x].dtype==object):
        data1[x]=le.fit_transform(data1[x]) 


# In[12]:


data1.info()


# In[13]:


#Data cleaning:outliers
sns.boxplot(data1["MonthlyIncome"])


# In[14]:


data1["MonthlyIncome"].quantile(0.92)


# In[15]:


data1["MonthlyIncome"]=np.where(data1["MonthlyIncome"]>16368.8,16368.8,
            data1["MonthlyIncome"])
sns.boxplot(data1["MonthlyIncome"]) 


# In[16]:


X=data1.drop(["Attrition"],axis=1)
Y=data1["Attrition"]


# In[17]:


#Feature Selection
import statsmodels.api as sm
X_con=sm.add_constant(X)


# In[18]:


model=sm.Logit(Y,X_con).fit()
print(model.summary())


# In[19]:


X1=X_con.drop(["BusinessTravel"],axis=1)
model1=sm.Logit(Y,X1).fit()
print(model1.summary())


# In[20]:


X2=X1.drop(["Education"],axis=1)
model2=sm.Logit(Y,X2).fit()
print(model2.summary())


# In[21]:


X3=X2.drop(["HourlyRate"],axis=1)
model3=sm.Logit(Y,X3).fit()
print(model3.summary())


# In[22]:


X4=X3.drop(["MonthlyRate"],axis=1)
model4=sm.Logit(Y,X4).fit()
print(model4.summary())


# In[23]:


X5=X4.drop(["PerformanceRating"],axis=1)
model5=sm.Logit(Y,X5).fit()
print(model5.summary())


# In[24]:


X6=X5.drop(["JobLevel"],axis=1)
model6=sm.Logit(Y,X6).fit()
print(model6.summary())


# In[25]:


X7=X6.drop(["PercentSalaryHike"],axis=1)
model7=sm.Logit(Y,X7).fit()
print(model7.summary())


# In[26]:


X8=X7.drop(["EducationField"],axis=1)
model8=sm.Logit(Y,X8).fit()
print(model8.summary())


# In[27]:


X9=X8.drop(["JobRole"],axis=1)
model9=sm.Logit(Y,X9).fit()
print(model9.summary())


# In[28]:


X10=X9.drop(["DailyRate"],axis=1)
model10=sm.Logit(Y,X10).fit()
print(model10.summary())


# In[29]:


X11=X10.drop(["StockOptionLevel"],axis=1)
model11=sm.Logit(Y,X11).fit()
print(model11.summary())


# In[30]:


X12=X11.drop(["TotalWorkingYears"],axis=1)
model12=sm.Logit(Y,X12).fit()
print(model12.summary())


# In[31]:


final=X12


# In[32]:


from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(final.values, i) for i in range(final.shape[1])]
vif["features"] = final.columns

#Inspect VIF Factors
vif.round(1)


# In[33]:


final=final.drop(["const"],axis=1)
final.info()


# In[34]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(final,Y,
                                test_size=0.30,random_state=550)


# In[35]:


#Logistic Regression


# In[36]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#answer prediction[0,1]
print(y_pred)


# In[37]:


from sklearn.metrics import accuracy_score

print("Logistic Regression Model Accuracy :",accuracy_score(y_test,y_pred))


# In[38]:


y_prob=model.predict_proba(x_test)
print(y_prob)
var=np.where(y_prob[0:,0:1]>0.50,0,1).flatten()
print(var)
print(accuracy_score(y_test,var))


# In[39]:


thes=[]
acc=[]
for i in range(1,101):
    tv=i/100
    var=np.where(y_prob[0:,0:1]>tv,0,1).flatten()
    acc.append(accuracy_score(y_test,var))
    thes.append(tv)
    
#acc vs thes dataframe

df=pd.DataFrame({"Threshold":thes,
                 "accuarcy":acc})
df.iloc[30:60,0:]


# In[40]:


'''Model threshold value=0.44 acc=0.888889'''
y_prob=model.predict_proba(x_test)
print(y_prob)
var=np.where(y_prob[0:,0:1]>0.44,0,1).flatten()
print(var)
logistic=accuracy_score(y_test,var)
print(logistic)


# In[41]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[42]:


from sklearn.metrics import r2_score,mean_squared_error
print("r2_score=",r2_score(y_test,y_pred))
print("mean squared=",mean_squared_error(y_test,y_pred))
#train/test/validation~ error/acc/score
train_set=model.score(x_train,y_train)
print("train set score=",train_set)
test_set=model.score(x_test,y_test)
print("test set score=",test_set)
X_val=x_train.sample(n = 441, random_state = 42)
Y_val=y_train.sample(n = 441, random_state = 42)
val_set=model.score(X_val,Y_val)
print("val=",val_set)

score=np.array([train_set, test_set, val_set])
bias=np.mean(score)
print("bias=",bias)


var=np.var(score)
print("variance=",var)


# In[43]:


#KNN Model
from sklearn.neighbors import KNeighborsClassifier
model1=KNeighborsClassifier(n_neighbors=4)

#model train
model1.fit(x_train,y_train)

#model test
KNN_pred=model1.predict(x_test)
print(KNN_pred)

from sklearn.metrics import accuracy_score
KNN=accuracy_score(y_test,KNN_pred)
print("KNN Model accuarcy: ",KNN)


# In[44]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, KNN_pred))
print(classification_report(y_test, KNN_pred))


# In[45]:


from sklearn.metrics import r2_score,mean_squared_error
print("r2_score=",r2_score(y_test,KNN_pred))
print("mean squared=",mean_squared_error(y_test,KNN_pred))
#train/test/validation~ error/acc/score
train_set=model1.score(x_train,y_train)
print("train set score=",train_set)
test_set=model1.score(x_test,y_test)
print("test set score=",test_set)
X_val=x_train.sample(n = 441, random_state = 42)
Y_val=y_train.sample(n = 441, random_state = 42)
val_set=model1.score(X_val,Y_val)
print("val=",val_set)

score=np.array([train_set, test_set, val_set])
bias=np.mean(score)
print("bias=",bias)


var=np.var(score)
print("variance=",var)


# In[46]:


#Random Forest

from sklearn.ensemble import RandomForestClassifier
modelrf=RandomForestClassifier()
modelrf.fit(x_train,y_train)
pred=modelrf.predict(x_test)
from sklearn.metrics import accuracy_score
RandomForest=accuracy_score(y_test,pred)
print("Random Forest accuracy=",accuracy_score(y_test,pred))


# In[47]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# In[48]:


from sklearn.metrics import r2_score,mean_squared_error
print("r2_score=",r2_score(y_test,pred))
print("mean squared=",mean_squared_error(y_test,pred))
#train/test/validation~ error/acc/score
train_set=modelrf.score(x_train,y_train)
print("train set score=",train_set)
test_set=modelrf.score(x_test,y_test)
print("test set score=",test_set)
X_val=x_train.sample(n = 441, random_state = 42)
Y_val=y_train.sample(n = 441, random_state = 42)
val_set=modelrf.score(X_val,Y_val)
print("val=",val_set)

score=np.array([train_set, test_set, val_set])
bias=np.mean(score)
print("bias=",bias)


var=np.var(score)
print("variance=",var)


# In[49]:


#Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
pred_dt=dt.predict(x_test)
from sklearn.metrics import accuracy_score
DecisionTree=accuracy_score(y_test,pred_dt)
print("Decision tree model accuracy=",DecisionTree)


# In[50]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, pred_dt))
print(classification_report(y_test, pred_dt))


# In[51]:


from sklearn.metrics import r2_score,mean_squared_error
print("r2_score=",r2_score(y_test,pred_dt))
print("mean squared=",mean_squared_error(y_test,pred_dt))
#train/test/validation~ error/acc/score
train_set=dt.score(x_train,y_train)
print("train set score=",train_set)
test_set=dt.score(x_test,y_test)
print("test set score=",test_set)
X_val=x_train.sample(n = 441, random_state = 42)
Y_val=y_train.sample(n = 441, random_state = 42)
val_set=dt.score(X_val,Y_val)
print("val=",val_set)

score=np.array([train_set, test_set, val_set])
bias=np.mean(score)
print("bias=",bias)


var=np.var(score)
print("variance=",var)


# In[52]:


#NaiveBayes
from sklearn.naive_bayes import GaussianNB

from sklearn.decomposition import PCA
pca=PCA(n_components=18)
X_train=pca.fit_transform(x_train)
X_test=pca.transform(x_test)

GNB=GaussianNB()

#model training
GNB.fit(X_train,y_train)


y_predict=GNB.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score

NaiveBayes_Model=accuracy_score(y_test,y_predict)
print(('accuracy score: '),NaiveBayes_Model)


# In[53]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))


# In[54]:


from sklearn.metrics import r2_score,mean_squared_error
print("r2_score=",r2_score(y_test,y_predict))
print("mean squared=",mean_squared_error(y_test,y_predict))
#train/test/validation~ error/acc/score
train_set=GNB.score(x_train,y_train)
print("train set score=",train_set)
test_set=GNB.score(x_test,y_test)
print("test set score=",test_set)
X_val=x_train.sample(n = 441, random_state = 42)
Y_val=y_train.sample(n = 441, random_state = 42)
val_set=GNB.score(X_val,Y_val)
print("val=",val_set)

score=np.array([train_set, test_set, val_set])
bias=np.mean(score)
print("bias=",bias)


var=np.var(score)
print("variance=",var)


# In[55]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf', degree=8)
svclassifier.fit(x_train, y_train)
y_preds = svclassifier.predict(x_test)
from sklearn.metrics import accuracy_score
SVC=accuracy_score(y_test,y_preds)
print("SVC accuracy=",SVC)


# In[56]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_preds))
print(classification_report(y_test, y_preds))


# In[57]:


from sklearn.metrics import r2_score,mean_squared_error
print("r2_score=",r2_score(y_test,y_preds))
print("mean squared=",mean_squared_error(y_test,y_preds))
#train/test/validation~ error/acc/score
train_set=svclassifier.score(x_train,y_train)
print("train set score=",train_set)
test_set=svclassifier.score(x_test,y_test)
print("test set score=",test_set)
X_val=x_train.sample(n = 441, random_state = 42)
Y_val=y_train.sample(n = 441, random_state = 42)
val_set=svclassifier.score(X_val,Y_val)
print("val=",val_set)

score=np.array([train_set, test_set, val_set])
bias=np.mean(score)
print("bias=",bias)


var=np.var(score)
print("variance=",var)


# In[58]:


y=[logistic*100,RandomForest*100,NaiveBayes_Model*100,SVC*100,KNN*100,DecisionTree*100]


# In[59]:


print(y)


# In[60]:


x=('logistic','RandomForest','NaiveBayes','SVC','KNN','Decision Tree')


fig = plt.figure(figsize =(10, 7))
plt.bar(x, y,width = 0.4)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Various Models')
plt.show()


# In[65]:


#Features that contribute the most in attrition
feat_importances = pd.Series(modelrf.feature_importances_, index=final.columns)
feat_importances = feat_importances.nlargest(20)
feat_importances.plot(kind='barh')


# In[ ]:




