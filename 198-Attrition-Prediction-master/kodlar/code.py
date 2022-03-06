# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 19:25:14 2017

@author: mahima.gupta
"""
from sklearn.metrics import accuracy_score
import pandas as pd 
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import collections


def labelEncoding(data,var_mod):
    le = LabelEncoder()   
    for i in var_mod:
        data[i] = le.fit_transform(data[i])
    return data,le

#############################
#reading the csv file provided
################################

data=pd.read_csv("1452762979_586__HR_Employee_Attrition_Data.csv")
data_copy=data

byAttrition = data.groupby('Attrition')

print(byAttrition.describe())

c=collections.Counter(data['Attrition'])
print("count of Yes and No in attrition is:")
print ("c")

####unbalanced classes in attrition#####
print(data.describe())
from sklearn import model_selection
from sklearn.model_selection import train_test_split

numerical_columns = data.select_dtypes(include=['int64']).columns

############################################################
#converting all the numerical columns to categorical
###########################################################

#Qunatiles is basically is a function which divides total range into number of qunatiles specified
for x in numerical_columns:
    #print x
    try:
        data[x] = pd.qcut(data[x], 5, labels=["very low", "low", "medium","high","very high"])
    except Exception as e:
        if len(data[x].unique()) > 5 :
           data[x] = pd.cut(data[x], 5, labels=["very low", "low", "medium","high","very high"]) 
#           print("I have quantile problem\n")
        else: 
            #For these I will consider numerical values itself as a feature 
            print(x,e,"\n")           
            

categorical_columns=data.select_dtypes(include=['category']).columns
print(categorical_columns)
print(data['MonthlyIncome'])

object_columns=data.select_dtypes(include=['object']).columns
print(object_columns)
data,le=labelEncoding(data,categorical_columns)
data,le=labelEncoding(data,object_columns)
#train,test=divideTrainTest(data)

#Normalization not needed as all the columns are categorical#

#lets analyse attrition on basis of sex
%matplotlib
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
# le = LabelEncoder()
# le.fit(data['Attrition'])
# print(le.transform(data['Attrition'])) 
attrition_yes = data[data['Attrition']==1]['Gender'].value_counts()
attrition_no = data[data['Attrition']==0]['Gender'].value_counts()
df = pd.DataFrame([attrition_yes,attrition_no])
print("\nattrition_yes:\n ",attrition_yes)
print("\nattrition_no: \n",attrition_no)
df.index = ['attrition_yes','attrition_no']
df.plot(kind='bar',stacked=True, figsize=(15,8),label = ['Male','Female'])    #red and blue are default colors
print(data['Gender'])
#print(le.inverse_transform(data['Attrition']))

#lets analyse attrition on basis of Distance From Home
attrition_yes = data[data['Attrition']==1]['DistanceFromHome'].value_counts()
attrition_no = data[data['Attrition']==0]['DistanceFromHome'].value_counts()
df = pd.DataFrame([attrition_yes,attrition_no])
df.index = ['attrition_yes','attrition_no']
df.plot(kind='bar',stacked=True, figsize=(15,8))
print(data['DistanceFromHome'])
#Graph implies people living closeby are less likely to attrition

x=data.drop('Attrition',axis=1)
x=x.drop('EmployeeNumber',axis=1)
y=data['Attrition']

##########################
##detecting outliers######
##########################
import matplotlib.pyplot as plt
plt.ylim(0, 10)
data.boxplot(column=None, by=None, ax=None, fontsize=None, rot=0, grid=True, figsize=None, layout=None, return_type=None)

#####################
##removing outliers
#####################

for i in data.columns:
    #print(i,data[i].mean())
    data = data[np.abs(data[i]-data[i].mean())<=(2*data[i].std())]
    Outliers = data[np.abs(data[i]-data[i].mean())>=(2*data[i].std())]
    print(data.shape)

###outliers have been removed##
plt.ylim(0, 10)
data.boxplot(column=None, by=None, ax=None, fontsize=None, rot=0, grid=True, figsize=None, layout=None, return_type=None)


##########################
#dividing data into train and test
##########################

validation_size = 0.20
seed = 7
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=validation_size, random_state=seed)
# x_train,y_train=numpyMatrix(train)
# x_test,y_test=numpyMatrix(test) 

######### Random Forest Model Fitting ###########
##Random Forest comes out to be the best model out of all the appplied models
print("Applying Random Forest")
sample_leaf_options = [1,5,10]
for leaf_size in sample_leaf_options :
    model= RandomForestClassifier(n_estimators = 200, oob_score = 'TRUE', n_jobs = -1,random_state =50,max_features = "auto", min_samples_leaf = leaf_size)
    #random_state =50 is to make each time outcome consistent across calls
    model.fit(x_train,y_train)    
    y_pred= model.predict(x_test)
    conf_arr=confusion_matrix(y_test,y_pred)
    print(conf_arr)
    accuracy = accuracy_score(y_test,y_pred)
    print("Random Forest accuracy is:",accuracy)
    print("f1 score is: ",f1_score(y_test,y_pred))
    print("Cohen's kappa ",cohen_kappa_score(y_test,y_pred))
    print ("AUC - ROC : ", roc_auc_score(y_test,y_pred))
    
le.inverse_transform(y_pred)    

#calculatig the importance of features-a characterstic of random forest##
features = pd.DataFrame()
features['feature'] = x.columns
features['importance'] = model.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
#plotting the importance
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(20, 20))

#############################
#visualizing the confusion matrix that is the results
#################################

norm_conf = []
for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                interpolation='nearest')

width, height = conf_arr.shape

for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')

cb = fig.colorbar(res)
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
plt.xticks(range(width), alphabet[:width])
plt.yticks(range(height), alphabet[:height])
plt.savefig('confusion_matrix.png', format='png')


print("Applying Decision Tree")
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
y_predict=clf.predict(x_test)
cnf_matrix = confusion_matrix(y_predict, y_test)
print(cnf_matrix)
accuracy = accuracy_score(y_test,y_pred)
print("Decision Tree accuracy is:",accuracy)
print("f1 score is: ",f1_score(y_test,y_pred))
print("Cohen's kappa ",cohen_kappa_score(y_test,y_pred))
print ("AUC - ROC : ", roc_auc_score(y_test,y_pred))

from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier()
clf.fit(x_train, y_train)
y_predict=clf.predict(x_test)
cnf_matrix = confusion_matrix(y_predict, y_test)
print(cnf_matrix)
accuracy = accuracy_score(y_test,y_pred)
print("KNN accuracy is:",accuracy)
print("f1 score is: ",f1_score(y_test,y_pred))
print("Cohen's kappa ",cohen_kappa_score(y_test,y_pred))
print ("AUC - ROC : ", roc_auc_score(y_test,y_pred))

from sklearn.svm import SVC
# Create SVM classification object 
clf=SVC()
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
clf.fit(x_train, y_train)
y_predict=clf.predict(x_test)
cnf_matrix = confusion_matrix(y_predict, y_test)
print(cnf_matrix)
accuracy = accuracy_score(y_test,y_pred)
print("SVM accuracy is:",accuracy)
print("f1 score is: ",f1_score(y_test,y_pred))
print("Cohen's kappa ",cohen_kappa_score(y_test,y_pred))
print ("AUC - ROC : ", roc_auc_score(y_test,y_pred))

##using naive bayes##
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_train, y_train)
y_predict=clf.predict(x_test)
cnf_matrix = confusion_matrix(y_predict, y_test)
print(cnf_matrix)
accuracy = accuracy_score(y_test,y_pred)
print("Naive Bayes accuracy is:",accuracy)
print("f1 score is: ",f1_score(y_test,y_pred))
print("Cohen's kappa ",cohen_kappa_score(y_test,y_pred))
print ("AUC - ROC : ", roc_auc_score(y_test,y_pred))

###using logistic regression##
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(x_train, y_train)
y_predict=clf.predict(x_test)
cnf_matrix = confusion_matrix(y_predict, y_test)
print(cnf_matrix)
accuracy = accuracy_score(y_test,y_pred)
print("Logistic Regression accuracy is: ",accuracy)
print("f1 score is: ",f1_score(y_test,y_pred))
print("Cohen's kappa ",cohen_kappa_score(y_test,y_pred))
print ("AUC - ROC : ", roc_auc_score(y_test,y_pred))
