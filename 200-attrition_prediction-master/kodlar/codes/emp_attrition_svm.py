# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 17:45:47 2019

@author: ashishkr568
"""

# This code is for building a SVM model on the HR Attrition dataset

# Import Required packages and custom functions
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
#import feather
import pickle
#from sklearn.preprocessing import LabelEncoder
import joblib
#from sklearn.ensemble import RandomForestClassifier
# Import accuarcy metrices
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import recall_score, accuracy_score, f1_score
from sklearn.metrics import classification_report
#from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV, cross_val_score
#from sklearn.metrics import make_scorer

# Import custom functions
sys.path.insert(0, './functions/')
#from ak_generic_fun import dup_col_rows, get_null, high_corr_sets, rem_high_corr
#from ak_plotting_fun import plt_numeric_categorical_cols, plt_corr_mat_heatmap
from ak_plotting_fun import roc_auc_curve_plot, label_and_plot_confusion_matrix

np.random.seed(42)

model_name='SVM'

inp_data_loc=r"./input"

# Define input and output data location
out_data_loc= r"./output"+'/'+ model_name


# Create a folder to store RF model data
if not os.path.exists(out_data_loc):
    os.makedirs(out_data_loc)


# Read Training and validation set data
model_dict=joblib.load('./output/model_dict')

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

################--------SVM Classifier Implementation------####################

##------------------Hyperplane tuning using GridSearch-------------------------#
## Create parameters for gridsearchcv
#grid_param_svc=[{'kernel':['rbf'],
#               'C':[1,10,100,1000],
#               'gamma':[1e-3,1e-4]},
#                {'kernel':['linear'],
#                 'C':[1,10,100,1000],
#                 'gamma':[1e-3,1e-4]},
#                 {'kernel':['sigmoid'],
#                  'C':[1,10,100,1000],
#                  'gamma':[1e-3,1e-4]},
#                 {'kernel':['poly'],
#                  'C':[1,10,100,1000],
#                  'gamma':[1e-3,1e-4], 
#                  'degree':[1,2,3]}]
#    
#
## Create a custom scoring function. 
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
#from sklearn.svm import SVC
#svc_classifier=SVC()
#
## Perform gridsearch for the above grid parameters
## We need to provide the desired scoring criteria and cv
#svc_grid_search=GridSearchCV(estimator=svc_classifier,
#                            param_grid=grid_param_svc,
#                            scoring='precision',
#                            cv=10,
#                            n_jobs=-1)
#
## Fit the above defined model with the training set, to get the best parameters 
#svc_grid_search.fit(x_train,y_train)
#
## Get the best parameters and save it in a file
#grid_cv_params_svc=svc_grid_search.best_params_
#grid_cv_params_svc= pd.DataFrame(grid_cv_params_svc, index=[1,])
#grid_cv_params_svc.to_csv(out_data_loc+'/'+"svc_Grid_serch_Params.csv", index=False)
#
##-----------------------------------------------------------------------------#


# Read Best parameters and test accuracy
grid_cv_params_svc=pd.read_csv(out_data_loc+"/"+"svc_Grid_serch_Params.csv")

# Import Packages
from sklearn.svm import SVC

# Build a support vector classifier
svclassifier= SVC(kernel = grid_cv_params_svc.kernel[0],
                  gamma = grid_cv_params_svc.gamma[0],
                  C = grid_cv_params_svc.C[0],
                  degree=grid_cv_params_svc.degree[0],
                  probability=True)

# Train classifier
attrition_svclassifier_model=svclassifier.fit(x_train,y_train)

# Predict on training set
y_train_pred_svc= attrition_svclassifier_model.predict(x_train)

# Predict on validation set
y_valid_pred_svc= attrition_svclassifier_model.predict(x_valid)

# Get Cross validation accuracy on the training data
scores = cross_val_score(attrition_svclassifier_model,x_train,y_train, cv=10)
cross_validation_accuracy=np.mean(scores)


#------------------------Training set accuracy--------------------------------#

# Get Confuaion Matrix and plot it
svc_confusion_matrix_train=label_and_plot_confusion_matrix(y_true=y_train,
                                                       y_pred=y_train_pred_svc, 
                                                       test_type="Training",
                                                       model_name='SVC',
                                                       out_loc=out_data_loc)


# Get classification report which consist of Precision, Recall and F1-Score
# in pandas dataframe, In case the result is not required in pandas dataframe
# remove output_dict parameter (it will return a string)
svc_classification_report_train=classification_report(y_true=y_train,
                                                     y_pred=y_train_pred_svc,
                                                     digits=3,
                                                     output_dict=True)

# Get classification report in pandas dataframe
svc_classification_report_train=pd.DataFrame(svc_classification_report_train).transpose()
svc_classification_report_train=svc_classification_report_train.round(3)

# roc curve for Training set
roc_auc_curve_plot(y_true= y_train,
                   y_pred= y_train_pred_svc,
                   classifier=attrition_svclassifier_model,
                   test_type='Training',
                   model_name='SVC',
                   x_true= x_train, 
                   out_loc=out_data_loc)



#------------------------Validation set accuracy------------------------------#
# Get Confuaion Matrix and plot it
svc_confusion_matrix_valid=label_and_plot_confusion_matrix(y_true=y_valid,
                                                       y_pred=y_valid_pred_svc, 
                                                       test_type="Validation",
                                                       model_name='SVC',
                                                       out_loc=out_data_loc)


# Get classification report which consist of Precision, Recall and F1-Score
# in pandas dataframe, In case the result is not required in pandas dataframe
# remove output_dict parameter (it will return a string)
svc_classification_report_valid=classification_report(y_true=y_valid,
                                                    y_pred=y_valid_pred_svc, 
                                                    digits=3,
                                                    output_dict=True)

# Get classification report in pandas dataframe
svc_classification_report_valid=pd.DataFrame(svc_classification_report_valid).transpose()
svc_classification_report_valid=svc_classification_report_valid.round(3)



# roc_auc curve for Test set
roc_auc_curve_plot(y_true= y_valid,
                   y_pred= y_valid_pred_svc,
                   classifier=attrition_svclassifier_model,
                   test_type='Validation',
                   model_name='SVC',
                   x_true= x_valid,
                   out_loc=out_data_loc)

# Prediction on Test Set

# Read blind dataset
blind_ds= pd.read_csv('./input/Attrition_Prediction_Blind_Set.csv')

# Get Dependent variable
dep_var_name="Attrition"

# Read label encoder used on the training data to encode test data
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
y_test_pred_svc= attrition_svclassifier_model.predict(x_test)


#------------------------Test set accuracy------------------------------#
# Get Confuaion Matrix and plot it
svc_confusion_matrix_test=label_and_plot_confusion_matrix(y_true=y_test,
                                                       y_pred=y_test_pred_svc, 
                                                       test_type="Test",
                                                       model_name='SVC',
                                                       out_loc=out_data_loc)


# Get classification report which consist of Precision, Recall and F1-Score
# in pandas dataframe, In case the result is not required in pandas dataframe
# remove output_dict parameter (it will return a string)
svc_classification_report_test=classification_report(y_true=y_test,
                                                    y_pred=y_test_pred_svc, 
                                                    digits=3,
                                                    output_dict=True)

# Get classification report in pandas dataframe
svc_classification_report_test=pd.DataFrame(svc_classification_report_test).transpose()
svc_classification_report_test=svc_classification_report_test.round(3)



# roc_auc curve for Test set
roc_auc_curve_plot(y_true= y_test,
                   y_pred= y_test_pred_svc,
                   classifier=attrition_svclassifier_model,
                   test_type='Test',
                   model_name='SVC',
                   x_true= x_test, 
                   out_loc=out_data_loc)


# Align data in output format
d_time= datetime.datetime.now().strftime("%D %H:%M:%S")
algorithm=model_name
# Training Set Details
train_set_accuracy=float(svc_classification_report_train.loc[svc_classification_report_train.index=='accuracy']['precision'])*100
train_set_f1_yes=float(svc_classification_report_train.loc[svc_classification_report_train.index=='1']['f1-score'])*100
train_set_f1_no=float(svc_classification_report_train.loc[svc_classification_report_train.index=='0']['f1-score'])*100
train_set_precision_yes=float(svc_classification_report_train.loc[svc_classification_report_train.index=='1']['precision'])*100
train_set_precision_no=float(svc_classification_report_train.loc[svc_classification_report_train.index=='0']['precision'])*100
train_set_recall_yes=float(svc_classification_report_train.loc[svc_classification_report_train.index=='1']['recall'])*100
train_set_recall_no=float(svc_classification_report_train.loc[svc_classification_report_train.index=='0']['recall'])*100

# Cross Validation Accuracy
cross_validation_accuracy=round(cross_validation_accuracy*100,3)

# Validation Set Details
valid_set_accuracy=float(svc_classification_report_valid.loc[svc_classification_report_valid.index=='accuracy']['precision'])*100
valid_set_f1_yes=float(svc_classification_report_valid.loc[svc_classification_report_valid.index=='1']['f1-score'])*100
valid_set_f1_no=float(svc_classification_report_valid.loc[svc_classification_report_valid.index=='0']['f1-score'])*100
valid_set_precision_yes=float(svc_classification_report_valid.loc[svc_classification_report_valid.index=='1']['precision'])*100
valid_set_precision_no=float(svc_classification_report_valid.loc[svc_classification_report_valid.index=='0']['precision'])*100
valid_set_recall_yes=float(svc_classification_report_valid.loc[svc_classification_report_valid.index=='1']['recall'])*100
valid_set_recall_no=float(svc_classification_report_valid.loc[svc_classification_report_valid.index=='0']['recall'])*100

# Test Set Accuracy
test_set_accuracy=float(svc_classification_report_test.loc[svc_classification_report_test.index=='accuracy']['precision'])*100
test_set_f1_yes=float(svc_classification_report_test.loc[svc_classification_report_test.index=='1']['f1-score'])*100
test_set_f1_no=float(svc_classification_report_test.loc[svc_classification_report_test.index=='0']['f1-score'])*100
test_set_precision_yes=float(svc_classification_report_test.loc[svc_classification_report_test.index=='1']['precision'])*100
test_set_precision_no=float(svc_classification_report_test.loc[svc_classification_report_test.index=='0']['precision'])*100
test_set_recall_yes=float(svc_classification_report_test.loc[svc_classification_report_test.index=='1']['recall'])*100
test_set_recall_no=float(svc_classification_report_test.loc[svc_classification_report_test.index=='0']['recall'])*100

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
