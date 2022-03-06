#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:23:39 2019

@author: expert
"""

import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import pickle

def one_hot_encoding(input_file):
    
    global temp,new_dataframe
    temp = input_file.select_dtypes(include="object")
    print(temp.columns)
    for row in temp.columns:
        new_dataframe = pandas.get_dummies(temp[row], prefix= "category_")
        input_file = pandas.concat([input_file, new_dataframe], axis=1) 
        del input_file[row]
    print(len(input_file.columns))
    return input_file

def model_building(algorithms,predictors,target_variable):
    
    scores =[]
    names=[]
    X_train,X_test,Y_train,Y_test = model_selection.train_test_split(predictors,target_variable,test_size=0.2, random_state =0)
    #cross validation ( 10 k folds)
    for name,algo in algorithms:
        k_fold = model_selection.KFold(n_splits = 10, random_state = 0)
        cvResults = model_selection.cross_val_score(algo, X_train, Y_train,cv = k_fold, scoring ='accuracy')  
        scores.append(cvResults) 
        names.append(name) 
        print(str(name)+' : '+str(cvResults.mean())) 
        
def evaluation_roc_curve(algorithms,predictors,target_variable):
    
    X_train,X_test,Y_train,Y_test = model_selection.train_test_split(predictors,target_variable,test_size=0.2, random_state =0)
    
    for name,algo in algorithms:
        mod = algo
        model =  mod.fit(X_train,Y_train)
        # calculate the fpr and tpr for all thresholds of the classification
        probs = model.predict_proba(X_test)
        preds = probs[:,1]
        fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
        roc_auc = metrics.auc(fpr, tpr)
        # method I: plt
        plt.title('Receiver Operating Characteristic for '+ name)
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
   
def model_saving(predictors,target_variable):
    
    X_train,X_test,Y_train,Y_test = model_selection.train_test_split(predictors,target_variable,test_size=0.2, random_state =0)
    #Logistic Regression
    logclf =  LogisticRegression()
    final_model = logclf.fit(X_train,Y_train)   
    #Model name
    filename = 'Employee_Attrition_Model.sav'
    pickle.dump(final_model, open(filename, 'wb'))
    
    
if __name__ == '__main__':

    #Define the path for reading the data
    read_path = "/home/expert/Documents/Self_Learning/Employee_Attrition_transformed_data_2.csv"
    
    #reading the transformed data    
    input_data = pandas.read_csv("Employee_Attrition_transformed_data_2.csv")
        
    #oen hot encoding
    final_input_data = one_hot_encoding(input_data)
    
    #list of algorithms for cross validation
    algorithms = []
    algorithms.append(('Logisitic Regression', LogisticRegression())) 
    algorithms.append(('RandomForest Classifier', RandomForestClassifier())) 
    algorithms.append(('Decision Tree Classifier', DecisionTreeClassifier())) 
    algorithms.append(('AdaBoost Classifier', AdaBoostClassifier())) 
    
    target_variable = final_input_data['Attrition']
    predictors  = final_input_data.copy()
    del predictors['Attrition']
    
    #cross validation
    model_building(algorithms, predictors,target_variable)
    
    #evaluation using ROC curve
    evaluation_roc_curve(algorithms,predictors,target_variable)
    
    #saving the model
    model_saving(predictors,target_variable)
