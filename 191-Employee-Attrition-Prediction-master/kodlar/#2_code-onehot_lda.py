# importing essential libraries
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score as asc
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# read input data using pandas
df_train = pd.read_csv("/kaggle/input/ee-769-assignment1/train.csv")
df_test = pd.read_csv("/kaggle/input/ee-769-assignment1/test.csv")

# defining data preprocessing function
def df_operations(df):
    col_names = np.array(df.columns)
    if 'Attrition' in col_names:
        features = df.drop(["Attrition"], axis=1)
        attrition = df["Attrition"]
    else:
        features = df
        attrition = 0
    
    # dropping less important features
    ID = features["ID"]
    df_fs_1 = features.drop(["EmployeeCount","EmployeeNumber","ID"], axis=1)
    
    # changing features with dtype-'objects' into categories
    df_fs_1['BusinessTravel'] = df_fs_1['BusinessTravel'].astype('category')
    df_fs_1['Department'] = df_fs_1['Department'].astype('category')
    df_fs_1['EducationField'] = df_fs_1['EducationField'].astype('category')
    df_fs_1['Gender'] = df_fs_1['Gender'].astype('category')
    df_fs_1['JobRole'] = df_fs_1['JobRole'].astype('category')
    df_fs_1['MaritalStatus'] = df_fs_1['MaritalStatus'].astype('category')
    df_fs_1['OverTime'] = df_fs_1['OverTime'].astype('category')
    df_fs_1['JobRole'] = df_fs_1['JobRole'].astype('category')

    # One hot encoding for some important features
    df_fs_2 = df_fs_1.copy()
    df_fs_3 = pd.get_dummies(df_fs_2, columns=['BusinessTravel'], prefix = ['b_trav']) #one hot for business travel
    df_fs_3 = pd.get_dummies(df_fs_3, columns=['Department'], prefix = ['dept']) #one hot for department
    df_fs_3 = pd.get_dummies(df_fs_3, columns=['EducationField'], prefix = ['edu']) #one hot for education
    df_fs_3 = pd.get_dummies(df_fs_3, columns=['JobRole'], prefix = ['job']) #one hot for job
    df_fs_3 = pd.get_dummies(df_fs_3, columns=['MaritalStatus'], prefix = ['marr']) #one hot for marital status
    df_fs_3 = pd.get_dummies(df_fs_3, columns=['Gender'], prefix = ['gend']) #one hot for gender
    df_fs_3 = pd.get_dummies(df_fs_3, columns=['OverTime'], prefix = ['ot']) #one hot for over time

    features_ex = df_fs_3.copy()
    
    return(features_ex, attrition, ID)

# passing the train and test dataset through preprocessing function
X_train, y_train, ID_train = df_operations(df_train)
X_test, y_test, ID_test = df_operations(df_test)

# normalizing the above processed features
train_norm = X_train[X_train.columns[:]]
test_norm = X_test[X_test.columns[:]]

std_scale = preprocessing.StandardScaler().fit(train_norm)
X_train_norm = std_scale.transform(train_norm)
X_test_norm = std_scale.transform(test_norm)

# training and predicting using SVM model
classifier = LinearDiscriminantAnalysis()
classifier.fit(X_train_norm, y_train)

# predicting the test set result
y_pred_lda = classifier.predict(X_test_norm)

# saving the result into pandas df accoring to the desired format
result = pd.DataFrame({'ID': ID_test, 'Attrition': y_pred_lda})

# save the file in csv format
result.to_csv('result_onehot_LDA.csv', encoding='utf-8', index=False)