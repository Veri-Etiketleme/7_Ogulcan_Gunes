# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 19:04:00 2020

@author: ashishkr568
"""

# This code is for building a Neural Network on the Employee Attrition dataset

# Import Required packages and custom functions
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from tensorflow import keras
import pickle
import joblib

# Import accuarcy metrices
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Import custom functions
sys.path.insert(0, './functions/')
from ak_plotting_fun import  label_and_plot_confusion_matrix

np.random.seed(42)

model_name='NN'

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


# Bulid a neural network model

nn_model=tf.keras.Sequential([
        keras.layers.Input(shape=len(x_train.columns)),
        keras.layers.Dense(100,activation='relu'),
        keras.layers.Dense(100,activation='relu'),
        keras.layers.Dense(100,activation='relu'),
        keras.layers.Dense(len(y_train.unique()),activation='softmax')
        ])
    
# Compile model
nn_model.compile(loss='sparse_categorical_crossentropy',
                 optimizer='adam',
                 metrics=['acc'])


# Train model
history = nn_model.fit(x_train, np.asarray(y_train), batch_size=64, epochs=50, verbose=1, validation_split=0.2)

# Predict on Training Set
y_train_pred_nn= np.argmax(nn_model.predict(x_train), axis=1)

# Predict on Validation Set
y_valid_pred_nn= np.argmax(nn_model.predict(x_valid), axis=1)


#------------------------Training set accuracy--------------------------------#

# Get Confuaion Matrix and plot it
nn_confusion_matrix_train=label_and_plot_confusion_matrix(y_true=y_train,
                                                       y_pred=y_train_pred_nn, 
                                                       test_type="Training",
                                                       model_name='nn',
                                                       out_loc=out_data_loc)


# Get classification report which consist of Precision, Recall and F1-Score
# in pandas dataframe, In case the result is not required in pandas dataframe
# remove output_dict parameter (it will return a string)
nn_classification_report_train=classification_report(y_true=y_train,
                                                     y_pred=y_train_pred_nn,
                                                     digits=3,
                                                     output_dict=True)

# Get classification report in pandas dataframe
nn_classification_report_train=pd.DataFrame(nn_classification_report_train).transpose()
nn_classification_report_train=nn_classification_report_train.round(3)


#------------------------Validation set accuracy------------------------------#
# Get Confuaion Matrix and plot it
nn_confusion_matrix_valid=label_and_plot_confusion_matrix(y_true=y_valid,
                                                       y_pred=y_valid_pred_nn, 
                                                       test_type="Validation",
                                                       model_name='nn',
                                                       out_loc=out_data_loc)


# Get classification report which consist of Precision, Recall and F1-Score
# in pandas dataframe, In case the result is not required in pandas dataframe
# remove output_dict parameter (it will return a string)
nn_classification_report_valid=classification_report(y_true=y_valid,
                                                    y_pred=y_valid_pred_nn, 
                                                    digits=3,
                                                    output_dict=True)

# Get classification report in pandas dataframe
nn_classification_report_valid=pd.DataFrame(nn_classification_report_valid).transpose()
nn_classification_report_valid=nn_classification_report_valid.round(3)


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
y_test_pred_nn= np.argmax(nn_model.predict(x_test), axis=1)


#------------------------Test set accuracy------------------------------#
# Get Confuaion Matrix and plot it
nn_confusion_matrix_test=label_and_plot_confusion_matrix(y_true=y_test,
                                                       y_pred=y_test_pred_nn, 
                                                       test_type="Test",
                                                       model_name='nn',
                                                       out_loc=out_data_loc)


# Get classification report which consist of Precision, Recall and F1-Score
# in pandas dataframe, In case the result is not required in pandas dataframe
# remove output_dict parameter (it will return a string)
nn_classification_report_test=classification_report(y_true=y_test,
                                                    y_pred=y_test_pred_nn, 
                                                    digits=3,
                                                    output_dict=True)

# Get classification report in pandas dataframe
nn_classification_report_test=pd.DataFrame(nn_classification_report_test).transpose()
nn_classification_report_test=nn_classification_report_test.round(3)


# Align data in output format
d_time= datetime.datetime.now().strftime("%D %H:%M:%S")
algorithm=model_name
# Training Set Details
train_set_accuracy=float(nn_classification_report_train.loc[nn_classification_report_train.index=='accuracy']['precision'])*100
train_set_f1_yes=float(nn_classification_report_train.loc[nn_classification_report_train.index=='1']['f1-score'])*100
train_set_f1_no=float(nn_classification_report_train.loc[nn_classification_report_train.index=='0']['f1-score'])*100
train_set_precision_yes=float(nn_classification_report_train.loc[nn_classification_report_train.index=='1']['precision'])*100
train_set_precision_no=float(nn_classification_report_train.loc[nn_classification_report_train.index=='0']['precision'])*100
train_set_recall_yes=float(nn_classification_report_train.loc[nn_classification_report_train.index=='1']['recall'])*100
train_set_recall_no=float(nn_classification_report_train.loc[nn_classification_report_train.index=='0']['recall'])*100

# Cross Validation Accuracy
#cross_validation_accuracy=round(cross_validation_accuracy*100,3)
cross_validation_accuracy='NA'
# Validation Set Details
valid_set_accuracy=float(nn_classification_report_valid.loc[nn_classification_report_valid.index=='accuracy']['precision'])*100
valid_set_f1_yes=float(nn_classification_report_valid.loc[nn_classification_report_valid.index=='1']['f1-score'])*100
valid_set_f1_no=float(nn_classification_report_valid.loc[nn_classification_report_valid.index=='0']['f1-score'])*100
valid_set_precision_yes=float(nn_classification_report_valid.loc[nn_classification_report_valid.index=='1']['precision'])*100
valid_set_precision_no=float(nn_classification_report_valid.loc[nn_classification_report_valid.index=='0']['precision'])*100
valid_set_recall_yes=float(nn_classification_report_valid.loc[nn_classification_report_valid.index=='1']['recall'])*100
valid_set_recall_no=float(nn_classification_report_valid.loc[nn_classification_report_valid.index=='0']['recall'])*100

# Test Set Accuracy
test_set_accuracy=float(nn_classification_report_test.loc[nn_classification_report_test.index=='accuracy']['precision'])*100
test_set_f1_yes=float(nn_classification_report_test.loc[nn_classification_report_test.index=='1']['f1-score'])*100
test_set_f1_no=float(nn_classification_report_test.loc[nn_classification_report_test.index=='0']['f1-score'])*100
test_set_precision_yes=float(nn_classification_report_test.loc[nn_classification_report_test.index=='1']['precision'])*100
test_set_precision_no=float(nn_classification_report_test.loc[nn_classification_report_test.index=='0']['precision'])*100
test_set_recall_yes=float(nn_classification_report_test.loc[nn_classification_report_test.index=='1']['recall'])*100
test_set_recall_no=float(nn_classification_report_test.loc[nn_classification_report_test.index=='0']['recall'])*100

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



