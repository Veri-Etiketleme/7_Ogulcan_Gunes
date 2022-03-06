#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 16:22:35 2018

@author: wangzhe
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split



# Load the data in
hr = pd.read_csv("HRData.csv")

# Lets see what it looks like
print(hr.shape)
before_dedup = hr.shape[0]
hr.describe(include='all')
hr = hr[hr.Attrition != 'Termination']

# Check for missings
print(np.count_nonzero(hr.isnull().values))
print(hr.isnull().any())

# Check for duplicates
print(hr[hr.duplicated(keep=False)].shape)

# Strip whitespaces
hr = hr.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Check for conflicting types
hr.dtypes

# only 353 missing value, so decide to delete them
hr.dropna(axis=0, inplace=True)

# Get rid of all the duplicates
hr.drop_duplicates(inplace=True)
print("Duplicates Removed: " + str(before_dedup - hr.shape[0]))


#delete unrelated variables
hr.drop('StandardHours',axis=1, inplace=True)
hr.drop('EmployeeCount',axis=1, inplace=True)
hr.drop('EmployeeNumber',axis=1, inplace=True)
hr.drop('Over18',axis=1, inplace=True)
hr.drop('Application ID',axis=1, inplace=True)


#detect outliers
cols = ['JobSatisfaction', 'PercentSalaryHike','YearsSinceLastPromotion','TrainingTimesLastYear','HourlyRate', 'MonthlyIncome','DistanceFromHome','Age','DailyRate','Education','EnvironmentSatisfaction','JobInvolvement','JobLevel','DistanceFromHome','MonthlyRate','NumCompaniesWorked','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']
hr[cols] = hr[cols].applymap(np.int64) #change these columns into int64 datatype

def outliers(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

cols = ['JobSatisfaction', 'HourlyRate', 'MonthlyIncome','DistanceFromHome','Age','DailyRate','Education','EnvironmentSatisfaction','JobInvolvement','JobLevel','DistanceFromHome','MonthlyRate','NumCompaniesWorked','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']
hr[cols] = hr[cols].applymap(np.int64)

outliers(hr['Age'])
outliers(hr['DailyRate'])
outliers(hr['MonthlyIncome'])
outliers(hr['DistanceFromHome'])
outliers(hr['EnvironmentSatisfaction'])
outliers(hr['JobLevel'])
outliers(hr['JobSatisfaction'])
outliers(hr['NumCompaniesWorked'])
outliers(hr['PerformanceRating'])
outliers(hr['RelationshipSatisfaction'])
outliers(hr['TotalWorkingYears'])
outliers(hr['TrainingTimesLastYear'])
outliers(hr['WorkLifeBalance'])
outliers(hr['YearsAtCompany'])
outliers(hr['YearsInCurrentRole'])
outliers(hr['YearsSinceLastPromotion'])
outliers(hr['YearsWithCurrManager'])

#distribution graph
fig,ax = plt.subplots(3,3, figsize=(10,10))              
sns.distplot(hr['TotalWorkingYears'], ax = ax[0,0]) 
sns.distplot(hr['YearsAtCompany'], ax = ax[0,1]) 
sns.distplot(hr['DistanceFromHome'], ax = ax[0,2]) 
sns.distplot(hr['YearsInCurrentRole'], ax = ax[1,0]) 
sns.distplot(hr['YearsWithCurrManager'], ax = ax[1,1]) 
sns.distplot(hr['YearsSinceLastPromotion'], ax = ax[1,2]) 
sns.distplot(hr['PercentSalaryHike'], ax = ax[2,0]) 
sns.distplot(hr['JobSatisfaction'], ax = ax[2,1]) 
sns.distplot(hr['TrainingTimesLastYear'], ax = ax[2,2]) 
plt.show()

# Plot out the counts of OverTime
sns.factorplot("Attrition", col="OverTime", data=hr, kind="count", col_wrap=2, size=5)
plt.subplots_adjust(top=.8)
plt.suptitle('Attrition Counts by whether an Employee worked Over Time')


#Plot out the attrition rate and current employee rate group by Gender 
plt.figure(figsize=(12,8))
attrition_counts = (hr.groupby(['Gender'])['Attrition'].value_counts(normalize=True).rename('percentage').mul(100).reset_index().sort_values('Attrition'))
plt.title('Percent Distribution of Gender by Attrition')
sns.barplot(x="Gender", y="percentage", hue="Attrition", data=attrition_counts)




#correaltion between variables & Draw the heatmap 
num_hr = hr.select_dtypes(include=[np.number])

corr = num_hr._get_numeric_data().corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, center=0.0,
                      vmax = 1, square=True, linewidths=.5, ax=ax)
plt.savefig('corr-heat.png')
plt.show()



# Plot the distribution of age and monthlyincome by Attrition Factor
plt.figure(figsize=(12,8))
plt.title('Age distribution of Employees by Attrition')
sns.distplot(hr.Age[hr.Attrition == 'Voluntary Resignation'], bins = np.linspace(1,70,35))
sns.distplot(hr.Age[hr.Attrition == 'Current employee'], bins = np.linspace(1,70,35))
plt.legend(['Voluntary Resignation','Current Employees'])

plt.figure(figsize=(12,8))
plt.title('MonthlyIncome distribution of Employees by Attrition')
sns.distplot(hr.MonthlyIncome[hr.Attrition == 'Voluntary Resignation'], bins = np.linspace(1,70,35))
sns.distplot(hr.MonthlyIncome[hr.Attrition == 'Current employee'], bins = np.linspace(1,70,35))
plt.legend(['Voluntary Resignation','Current Employees'])

#transform category variables into invterval variables 
hr.replace({'Attrition':'Voluntary Resignation'},1,inplace=True)
hr.replace({'Attrition':'Current employee'},0,inplace=True)

hr.replace({'BusinessTravel':'Non-Travel'},1,inplace=True)
hr.replace({'BusinessTravel':'Travel_Rarely'},2,inplace=True)
hr.replace({'BusinessTravel':'Travel_Frequently'},3,inplace=True)

hr.replace({'Department':'Sales'},1,inplace=True)
hr.replace({'Department':'Research & Development'},2,inplace=True)
hr.replace({'Department':'Human Resources'},3,inplace=True)

hr.replace({'Gender':'Female'},0,inplace=True)
hr.replace({'Gender':'Male'},1,inplace=True)

hr.replace({'OverTime':'No'},0,inplace=True)
hr.replace({'OverTime':'Yes'},1,inplace=True)

hr.replace({'MaritalStatus':'Single'},1,inplace=True)
hr.replace({'MaritalStatus':'Married'},2,inplace=True)
hr.replace({'MaritalStatus':'Divorced'},3,inplace=True)


#get train_set and test_set
hr.dtypes
x=hr.select_dtypes(include=['int64'])
x.columns
y=hr['Attrition'] # y = dependent variable
x=x.drop('Attrition', axis=1) # x = independent variables set
x.columns
X_train,X_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=1)
#fit the model
model=LogisticRegression()
model.fit(X_train,y_train)

#test the model
predicted= model.predict(X_test)
print (predicted)
probs = model.predict_proba(X_test)
print (probs)
print (metrics.accuracy_score(y_test, predicted))
print (metrics.roc_auc_score(y_test, probs[:, 1]))

#confusion matrix
print (metrics.confusion_matrix(y_test, predicted))
cfm=metrics.confusion_matrix(y_test, predicted)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
classes_name=['NO','YES']
plot_confusion_matrix(cfm,classes = classes_name,
                      title='Attrition Confusion matrix(not normalized)')

#classification report 
print (metrics.classification_report(y_test, predicted,target_names = ['No','Yes']))
#" precision = TP/(TP+FP)"      
#" recall =TP/(TP+FN)  "
#" F1= 2*precision*recall/(recall+precision)" 
#" TP =True Positives  "
#" FP =Flase Positives " 
#" FN =Flase Negatives "
#" TN =True Negatives  "

#use random froest to make predict and compute the feature importances
rf = RandomForestClassifier(class_weight="balanced", n_estimators=500) 
rf.fit(X_train,y_train)
importances = rf.feature_importances_
names = hr.columns
importances, names = zip(*sorted(zip(importances, names)))

# Lets plot this
plt.figure(figsize=(12,8))
plt.barh(range(len(names)), importances, align = 'center')
plt.yticks(range(len(names)), names)
plt.xlabel('Importance of features')
plt.ylabel('Features')
plt.title('Importance of each feature')
plt.show()


#test the random forest
predicted2=rf.predict(X_test)
print (predicted)
probs2 = rf.predict_proba(X_test)
print (probs)
print (metrics.accuracy_score(y_test, predicted))
print (metrics.roc_auc_score(y_test, probs2[:, 1]))

scores = cross_val_score(rf, X_test, y_test, cv=10, scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(rf, X_test, y_test, cv=10, scoring='roc_auc')
print("ROC_AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
cfm2=metrics.confusion_matrix(y_test, predicted2)
plot_confusion_matrix(cfm2,classes = classes_name,
                      title='Attrition Confusion matrix(not normalized)')

