#File: classifiers.py
#Author: Austin Spadaro
#Description: Implements 6 different ML classifiers that are configured to be run against IBM employee attrition data.
#Notes: This program was developed against the Python runtime version 3.9.5

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import sklearn
from xgboost import XGBClassifier
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pickle

#Set the type of classifier to use.
modelType = "svm" #svm, Randomforest, Regression, Gaussian, DiscriminantAnalysis, DecisionTree, XGBoost

#Load our features into memory.
dataframe = pd.read_csv("Features\\features.csv")

#We need to randomize the records to avoid any selection bias.
dataframe = shuffle(dataframe)

#We need to remove the columns that do not tell us anything about our target labels.
dataframe = dataframe.drop(['EmployeeCount'], axis=1)
dataframe = dataframe.drop(['EmployeeNumber'], axis=1)
dataframe = dataframe.drop(['StandardHours'], axis=1)


dataframe.reset_index(inplace=True, drop=True)

#Separate our data based on the attrition label.
Att_Data = dataframe.loc[dataframe['Attrition'] == 1]
NoAtt_Data = dataframe.loc[dataframe['Attrition'] == 0]


attrition_num_train_yes = 100 #The number of records in our training set where attrition is yes.
attrition_num_train_no = 600  #The number of records in our training set where attrition is no.

attrition_num_test_yes = 50  #The number of records in our test set where attrition is yes.
attrition_num_test_no = 350  #The number of records in our test set where attrition is no.

#Take the attrition=yes records for the training data.
atd = Att_Data.sample(attrition_num_train_yes)
Att_Data = Att_Data.drop(atd.index)

#Take the attrition=no records for the training data.
ntd = NoAtt_Data.sample(attrition_num_train_no)
NoAtt_Data = NoAtt_Data.drop(ntd.index) #

#Combine our attrition=yes and attrition=no records into a single data set.
Train_Data = pd.concat([atd, ntd], axis=0) #.4 .26668 .7675

#Take the attrition=yes records for the test data.
atrd = Att_Data.sample(attrition_num_test_yes)
Att_Data = Att_Data.drop(atrd.index)

#Take the attrition=no records for the test data.
ntrd = NoAtt_Data.sample(attrition_num_test_no)
NoAtt_Data = NoAtt_Data.drop(ntrd.index)

#Combine our attrition=yes and attrition=no records into a single data set.
Test_Data = pd.concat([atrd, ntrd], axis=0)

#A module for scaling our data before we use it to train our svm.
scaler = StandardScaler()



#For the training records, seperate our labels from feature set.

Train_Data = shuffle(Train_Data)
Train_Data.reset_index(inplace=True, drop=True)

y_train = Train_Data['Attrition'].values
X_train = Train_Data.drop(['Attrition'], axis=1)

X_train = scaler.fit_transform(X_train) #Note: We are scaling our data after we have separated out our test/train data.


#For the testing records, seperate our labels from feature set.

Test_Data = shuffle(Test_Data)
Test_Data.reset_index(inplace=True, drop=True)

y_test = Test_Data['Attrition'].values
X_test = Test_Data.drop(['Attrition'], axis=1)

X_test = scaler.fit_transform(X_test) #Note: We are scaling our data after we have separated out our test/train data.

#The documentation for these model parameters can be found at https://scikit-learn.org
if modelType == "svm":
    model = svm.SVC(kernel='rbf', C=2, gamma='scale') #radial basis function(rbf) kernel.

elif modelType == "Randomforest":
    model = RandomForestClassifier(n_estimators=100)
    
elif modelType == "Regression":
    model = LogisticRegression(penalty='l2', C=1.5, dual=DiscriminantAnalysisFalse, fit_intercept=True, intercept_scaling=1, solver='lbfgs')
    
elif modelType == "Gaussian":
    model = GaussianNB(var_smoothing=.00005)
    
elif modelType == "DiscriminantAnalysis":
    model = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, n_components=None, store_covariance=True)
    
elif modelType == "DecisionTree":
    model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=6, min_weight_fraction_leaf=0.5)
    
elif modelType == "XGBoost":
    model = XGBClassifier(use_label_encoder=False, base_score=0.5, booster='gbtree', n_estimators=900, n_jobs=20, gamma=1, learning_rate=2)
    
else:
    print("Classifier model not selected.")
    quit()

#Train our model using the training data.
model.fit(X_train, y_train)

#We have a list of test cases in 'X_test'. Passing these in to model.predict() will return a list of predictions, one for each record in 'X_test'.
y_pred = model.predict(X_test)

#Uncomment this line to print the accuracy to the console.
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

cnf_matrix = confusion_matrix(y_test, y_pred)

#Print the confusion matrix to the console. 
print(cnf_matrix)

#Some settings for a graphical heat map.
#df_cm = pd.DataFrame(cnf_matrix, range(2), range(2))
#sn.set(font_scale=1.4)
#sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
scores = cross_val_score(model, X_train, y_train, cv=10)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
#Uncomment this line to display the confusion matrix heat map.
#plt.show()


