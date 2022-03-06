# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 13:51:42 2019

@author: ashishkr568
"""

# This code is for building a decision tree classifier on the HR Attrition dataset

# Import Required packages and custom functions
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import feather
#import pickle
#from sklearn.preprocessing import LabelEncoder
import joblib
# Import accuarcy metrices
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import recall_score, accuracy_score, f1_score
from sklearn.metrics import classification_report
#from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
#from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeClassifier
import datetime

np.random.seed(42)

# Import custom functions
sys.path.insert(0, './functions/')
#from ak_generic_fun import dup_col_rows, get_null, high_corr_sets, rem_high_corr
#from ak_plotting_fun import plt_numeric_categorical_cols, plt_corr_mat_heatmap
from ak_plotting_fun import roc_auc_curve_plot, label_and_plot_confusion_matrix


model_name='Decision_Tree'

inp_data_loc=r"./input/"

# Define input and output data location
out_data_loc= r"./output"+'/'+ model_name


# Create a folder to store RF model data
if not os.path.exists(out_data_loc):
    os.makedirs(out_data_loc)


# Read Training and validation set data
model_dict=joblib.load('./output'+'/'+'model_dict')

x_train=model_dict['x_train']
y_train=model_dict['y_train']

x_valid=model_dict['x_valid']
y_valid=model_dict['y_valid']


# As there is huge class imblalance, lets try to deal it using SMOTE
# SMOTE - Synthetic Minority Over Sampling Techniques.
# There are following other ways as well to deal with severe class imbalance
# 1. Synthesis of new minority class instance
# 2. Over sampling of minority class
# 3. Under Sampling of majority class
# 4. tweek the cost function to make misclassification of minority instances 
#    more important than misclassification of majority instances

## Check Data imbalance before oversampling
pd.value_counts(y_train).plot.bar()
plt.title('Attrition Distribution- Before Oversampling')
plt.xlabel('Employee Exit (0-N, 1-Y)')
plt.ylabel('Frequency')
plt.savefig(out_data_loc+'/'+"Attrition_Distribution- Before Oversampling.jpeg",bbox='tight')


from imblearn.over_sampling import SMOTE
smt=SMOTE()
x_train,y_train = smt.fit_sample(x_train,y_train)

# Check Data imbalance after oversampling
pd.value_counts(y_train).plot.bar()
plt.title('Attrition Distribution- After Oversampling')
plt.xlabel('Employee Exit (0-N, 1-Y)')
plt.ylabel('Frequency')
plt.savefig(out_data_loc+'/'+"Attrition_Distribution- After Oversampling.jpeg",bbox='tight')


#-------------------Hyperparameter tuning using GridSearchCV------------------#
## Create parameters for gridsearchcv
#grid_param_dtree={'criterion':['gini','entropy'],
#                  'splitter':['best'],
#                  'max_depth':[6,10,None],
#                  'min_samples_split':[2,5,10,25],
#                  'min_samples_leaf':[2,4,6],
#                  'random_state':[0]}
#
#
### Create a custom scoring function. 
### We will be using recall for the positive class.
##def custom_recall_fun(y,y_pred):
##    confusion_mat_temp=confusion_matrix(y,y_pred, labels=[1,0]).T
##    recall_positive=confusion_mat_temp[0,0]/(confusion_mat_temp[0,0]+confusion_mat_temp[1,0])
##    return recall_positive
##
### Use make scorer from sklearn.metrices to create a scoring function
##custom_scorer = make_scorer(score_func=custom_recall_fun, greater_is_better=True)
#
#
## Build a dummy classifier
#dtree_classifier=DecisionTreeClassifier()
#
## Perform gridsearch for the above grid parameters
## We need to provide the desired scoring criteria and cv
#dtree_grid_search=GridSearchCV(estimator=dtree_classifier,
#                            param_grid=grid_param_dtree,
#                            scoring='precision',
#                            return_train_score=True,
#                            cv=StratifiedKFold(n_splits=10))
#
## Fit the above defined model with the training set, to get the best parameters 
#dtree_grid_search.fit(x_train,y_train)
#
## Save gridsearch results for future reference
#cv_results=pd.DataFrame.from_dict(dtree_grid_search.cv_results_,orient='columns')
#
#
## Get the best parameters and save it in a file
#grid_cv_params_dtree=dtree_grid_search.best_params_
#grid_cv_params_dtree= pd.DataFrame(grid_cv_params_dtree, index=[1,])
#grid_cv_params_dtree.to_csv(out_data_loc+'/'+"D_Tree_Grid_serch_Params.csv", index=False)

#-----------------------------------------------------------------------------#


# Read Best parameters and test accuracy
grid_cv_params_dtree=pd.read_csv(out_data_loc+"/"+"D_Tree_Grid_serch_Params.csv")
grid_cv_params_dtree = grid_cv_params_dtree.where((pd.notnull(grid_cv_params_dtree)), None)

# Build a Decision tree classifier
dtree_classifier= DecisionTreeClassifier(criterion = grid_cv_params_dtree.criterion[0],
                                         splitter = grid_cv_params_dtree.splitter[0],
                                         max_depth = grid_cv_params_dtree.max_depth[0],
                                         min_samples_split = grid_cv_params_dtree.min_samples_split[0],
                                         min_samples_leaf=grid_cv_params_dtree.min_samples_leaf[0],
                                         random_state=0)


# Get Cross validation accuracy on the training data
scores = cross_val_score(dtree_classifier,x_train,y_train, cv=10)
cross_validation_accuracy=np.mean(scores)


# Train classifier
attrition_dtree_model=dtree_classifier.fit(x_train,y_train)

# Predict on training set
y_train_pred_dtree= attrition_dtree_model.predict(x_train)

# Predict on validation set
y_valid_pred_dtree= attrition_dtree_model.predict(x_valid)



# An employee is always a great asset to an organization. This tool will help
# help its HR team to check whether an employee is going to Resign or not.
# Once the probable employees who would resign are identified, the HR team can
# look for various options to retain them. The various options might be in 
# terms of compensation, good work life balance, travel options etc.


# True Positive - The tool correctly predicts that the employee is going to Resign
# False Positive- The tool predicts that the employee is going to resign which is not true
# False Negative- The tool predicts that the employee is not going to resign but resigns
# True Negative- The tool correctly predicts that the employee is not going to resign

# In the above case the cost of false negative is high. We need to limit the rate of 
# False negative, so a better accuracy metric will be the one wich deals with
# False Negative. We will consider Recall/ sensitivity, F1 Score,
# roc_auc score and finally auc score for model comparision.


#------------------------Training set accuracy--------------------------------#

# Get Confuaion Matrix and plot it
dtree_confusion_matrix_train=label_and_plot_confusion_matrix(y_true=y_train,
                                                       y_pred=y_train_pred_dtree, 
                                                       test_type="Training",
                                                       model_name='D-Tree',
                                                       out_loc=out_data_loc)


# Get classification report which consist of Precision, Recall and F1-Score
# in pandas dataframe, In case the result is not required in pandas dataframe
# remove output_dict parameter (it will return a string)
dtree_classification_report_train=classification_report(y_true=y_train,
                                                     y_pred=y_train_pred_dtree,
                                                     digits=3,
                                                     output_dict=True)

# Get classification report in pandas dataframe
dtree_classification_report_train=pd.DataFrame(dtree_classification_report_train).transpose()
dtree_classification_report_train=dtree_classification_report_train.round(3)

# roc curve for Training set
roc_auc_curve_plot(y_true= y_train,
                   y_pred= y_train_pred_dtree,
                   classifier=attrition_dtree_model,
                   test_type='Training',
                   model_name='D-Tree',
                   x_true= x_train, 
                   out_loc=out_data_loc)



#------------------------Validation set accuracy------------------------------#
# Get Confuaion Matrix and plot it
dtree_confusion_matrix_valid=label_and_plot_confusion_matrix(y_true=y_valid,
                                                       y_pred=y_valid_pred_dtree, 
                                                       test_type="Validation",
                                                       model_name='D-Tree',
                                                       out_loc=out_data_loc)


# Get classification report which consist of Precision, Recall and F1-Score
# in pandas dataframe, In case the result is not required in pandas dataframe
# remove output_dict parameter (it will return a string)
dtree_classification_report_valid=classification_report(y_true=y_valid,
                                                    y_pred=y_valid_pred_dtree, 
                                                    digits=3,
                                                    output_dict=True)

# Get classification report in pandas dataframe
dtree_classification_report_valid=pd.DataFrame(dtree_classification_report_valid).transpose()
dtree_classification_report_valid=dtree_classification_report_valid.round(3)



# roc_auc curve for Test set
roc_auc_curve_plot(y_true= y_valid,
                   y_pred= y_valid_pred_dtree,
                   classifier=attrition_dtree_model,
                   test_type='Validation',
                   model_name='D-Tree',
                   x_true= x_valid, 
                   out_loc=out_data_loc)


# Visualize Decision Tree
from sklearn import tree
import graphviz
# Note: Graphviz has to be installed on a system in case the tree has to be visualized
# It can be installed from the below location
# https://graphviz.gitlab.io/_pages/Download/Download_windows.html

# Add Graphviz folder location to PATH
os.environ["PATH"] += os.pathsep + r'C:\graphviz-2.38\release\bin'


dot_data = tree.export_graphviz(attrition_dtree_model, out_file=None, 
                                feature_names=x_train.columns.str.replace('&','_'),  
                                class_names=np.sort(y_train.unique()).astype(str),  
                                filled=True, rounded=True,  
                                special_characters=True)  
graph = graphviz.Source(dot_data)  

# Save plot. The plot is saved in pdf format
graph.render(out_data_loc+'/'+'Attrition_Tree')


# Prediction on Test Set

# Read blind dataset
blind_ds= pd.read_csv('./input/Attrition_Prediction_Blind_Set.csv')

# Get Dependent variable
dep_var_name="Attrition"

# Read label encoder used on the training data to encode test data
import pickle
le_att=pickle.load(open('output/'+'Attrition_encoder.pkl','rb'))
blind_ds.Attrition=le_att.transform(blind_ds.Attrition)

# Create categorical columns (Same as that of Training Set)
cat_col_list=model_dict['Category_Columns']
blind_ds=pd.get_dummies(blind_ds, columns=cat_col_list)


# Select only columns present in Training set
x_test=blind_ds[x_train.columns]

# Scale Test set with same weights as Training set
std_scalar=joblib.load("./output/std_scalar")

test_cols=x_test.columns
x_test=std_scalar.transform(x_test)
x_test=pd.DataFrame(x_test, columns=test_cols)


# Select dependent variable to check accuracy
y_test=blind_ds[dep_var_name]


# Predict on Test set
y_test_pred_dtree= attrition_dtree_model.predict(x_test)


#------------------------Test set accuracy------------------------------#
# Get Confuaion Matrix and plot it
dtree_confusion_matrix_test=label_and_plot_confusion_matrix(y_true=y_test,
                                                       y_pred=y_test_pred_dtree, 
                                                       test_type="Test",
                                                       model_name='D-Tree',
                                                       out_loc=out_data_loc)


# Get classification report which consist of Precision, Recall and F1-Score
# in pandas dataframe, In case the result is not required in pandas dataframe
# remove output_dict parameter (it will return a string)
dtree_classification_report_test=classification_report(y_true=y_test,
                                                    y_pred=y_test_pred_dtree, 
                                                    digits=3,
                                                    output_dict=True)

# Get classification report in pandas dataframe
dtree_classification_report_test=pd.DataFrame(dtree_classification_report_test).transpose()
dtree_classification_report_test=dtree_classification_report_test.round(3)



# roc_auc curve for Test set
roc_auc_curve_plot(y_true= y_test,
                   y_pred= y_test_pred_dtree,
                   classifier=attrition_dtree_model,
                   test_type='Test',
                   model_name='D-Tree',
                   x_true= x_test, 
                   out_loc=out_data_loc)

# Get feature importanance for decison tree
dtree_feature_imp=pd.concat([pd.DataFrame(x_train.columns),pd.DataFrame(dtree_classifier.feature_importances_)],axis=1)
dtree_feature_imp.columns=['Parameter_Name','Importance']
dtree_feature_imp=dtree_feature_imp.sort_values(by=["Importance"], ascending=False)
dtree_feature_imp=dtree_feature_imp.loc[dtree_feature_imp['Importance']>0,]
dtree_feature_imp.to_csv(out_data_loc+'/Dtree_feature_imp.csv', index=False)



# Align data in output format
d_time= datetime.datetime.now().strftime("%D %H:%M:%S")
algorithm=model_name
# Training Set Details
train_set_accuracy=float(dtree_classification_report_train.loc[dtree_classification_report_train.index=='accuracy']['precision'])*100
train_set_f1_yes=float(dtree_classification_report_train.loc[dtree_classification_report_train.index=='1']['f1-score'])*100
train_set_f1_no=float(dtree_classification_report_train.loc[dtree_classification_report_train.index=='0']['f1-score'])*100
train_set_precision_yes=float(dtree_classification_report_train.loc[dtree_classification_report_train.index=='1']['precision'])*100
train_set_precision_no=float(dtree_classification_report_train.loc[dtree_classification_report_train.index=='0']['precision'])*100
train_set_recall_yes=float(dtree_classification_report_train.loc[dtree_classification_report_train.index=='1']['recall'])*100
train_set_recall_no=float(dtree_classification_report_train.loc[dtree_classification_report_train.index=='0']['recall'])*100

# Cross Validation Accuracy
cross_validation_accuracy=round(cross_validation_accuracy*100,3)

# Validation Set Details
valid_set_accuracy=float(dtree_classification_report_valid.loc[dtree_classification_report_valid.index=='accuracy']['precision'])*100
valid_set_f1_yes=float(dtree_classification_report_valid.loc[dtree_classification_report_valid.index=='1']['f1-score'])*100
valid_set_f1_no=float(dtree_classification_report_valid.loc[dtree_classification_report_valid.index=='0']['f1-score'])*100
valid_set_precision_yes=float(dtree_classification_report_valid.loc[dtree_classification_report_valid.index=='1']['precision'])*100
valid_set_precision_no=float(dtree_classification_report_valid.loc[dtree_classification_report_valid.index=='0']['precision'])*100
valid_set_recall_yes=float(dtree_classification_report_valid.loc[dtree_classification_report_valid.index=='1']['recall'])*100
valid_set_recall_no=float(dtree_classification_report_valid.loc[dtree_classification_report_valid.index=='0']['recall'])*100

# Test Set Accuracy
test_set_accuracy=float(dtree_classification_report_test.loc[dtree_classification_report_test.index=='accuracy']['precision'])*100
test_set_f1_yes=float(dtree_classification_report_test.loc[dtree_classification_report_test.index=='1']['f1-score'])*100
test_set_f1_no=float(dtree_classification_report_test.loc[dtree_classification_report_test.index=='0']['f1-score'])*100
test_set_precision_yes=float(dtree_classification_report_test.loc[dtree_classification_report_test.index=='1']['precision'])*100
test_set_precision_no=float(dtree_classification_report_test.loc[dtree_classification_report_test.index=='0']['precision'])*100
test_set_recall_yes=float(dtree_classification_report_test.loc[dtree_classification_report_test.index=='1']['recall'])*100
test_set_recall_no=float(dtree_classification_report_test.loc[dtree_classification_report_test.index=='0']['recall'])*100

# Create output dataframe
out_dict={"DataTime":[d_time],
          "Algorithm":[algorithm],
          "Training_Set_Accuracy":[train_set_accuracy],
          "Training_Set_F1_Score_Yes":[train_set_f1_yes],
          "Training_Set_F1_Score_No":[train_set_f1_no],
          "Training_Set_Precision_Yes":[train_set_precision_yes],
          "Training_Set_Precision_No":[train_set_precision_no],
          "Training_Set_Recall_Yes":[train_set_recall_yes],
          "Training_Set_Recall_No":[train_set_recall_no],
          "Cross_Validation_Accuracy":[cross_validation_accuracy],
          "Validation_Set_Accuracy":[valid_set_accuracy],
          "Validation_Set_F1_Score_Yes":[valid_set_f1_yes],
          "Validation_Set_F1_Score_No":[valid_set_f1_no],
          "Validation_Set_Precision_Yes":[valid_set_precision_yes],
          "Validation_Set_Precision_No":[valid_set_precision_no],
          "Validation_Set_Recall_Yes":[valid_set_recall_yes],
          "Validation_Set_Recall_No":[valid_set_recall_no],
          "Test_Set_Accuracy":[test_set_accuracy],
          "Test_Set_F1_Score_Yes":[test_set_f1_yes],
          "Test_Set_F1_Score_No":[test_set_f1_no],
          "Test_Set_Precision_Yes":[test_set_precision_yes],
          "Test_Set_Precision_No":[test_set_precision_no],
          "Test_Set_Recall_Yes":[test_set_recall_yes],
          "Test_Set_Recall_No":[test_set_recall_no]
          }

out_df=pd.DataFrame.from_dict(out_dict)

# Append the result in existing accuracy nmetrices sheet
acc_met=pd.read_csv("./output/Accuracy_Metrices.csv")
acc_met=pd.concat([acc_met,out_df],axis=0)
acc_met.to_csv("./output/Accuracy_Metrices.csv",index=False)
