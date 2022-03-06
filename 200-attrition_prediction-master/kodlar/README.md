# Employee Attrition Prediction

This repository is for predicting whether an employee will stay of leave an organization based of provided features. 
Please refer the below link for more details on the dataset.
Link:- https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset

This project can act as a guide to someone who is new to the field of data science and wants to get a brief idea about data preprocessing steps and basic model development.
However this is not the exhaustive list of task performed for data pre-processing. The data preprocessing steps largly depends on the type of data you are working on.

## Algorithms Used
1. Decision Tree
2. Random Forest
3. XG Boost
4. Support Vector Machines (SVM)
5. Neural Network


## Getting Started
In order to use this project the user needs to clone it and then they need to create a project using any of the python IDE (preferably Spyder).
Once done the user can run the run the code and get the prediction for Employee attrition on the included dataset.


## Folder Structure
There are 4 folders in this project and below are their decription:

#### codes
This folder has the codes to perform the following tasks:

- Preprocess Data (hr_attrition_preprocess.py) - This code performs the following tasks:
    - Read Data
    - Check for missing values
    - Check for duplicate rows
    - Parameter Encoding
    - Remove highly correlated features
    - Train / Test split 
    - Feature scaling
    - Oversampling (SMOTE)
- Build a Decision Tree on the preprocessed data (hr_attrition_dtree.py)
- Build a Random forest model on the preprocessed data (hr_attrition_rf.py)
- Build a xgboost model on the preprocessed data (hr_attrition_xgb.py)
- Build a SVM model on the preprocessed data (hr_attrition_svm.py)
    
#### input
This folder contains the input/raw file used for this project.

#### functions
This folder contains functions used for plotting and preprocessing the data. Below are the functions included in it:

- **Preprocessing function (ak_generic_fun.py)**
    - Check and remove duplicate rows and columns
    - Get list of columns with missing values
    - Get list of highly correlated columns
    - Remove highly correlated columns based on correlation coeffecient provided by user
    - Extract all information from a datetime object
- **Plotting Function (ak_plotting_fun)**
    - Birds eye view of categorical and Numeric columns
    - Plot correlation matrix as heatmap
    - Plot ROC_AUC curve
    - Plot confusion matrix with labels

#### output
This folder will store all the generated outputs
    
**Note-** These files are created using python 3.7 in Spyder.

## Version

Initial version (v0)

As the accuracy improves or a new algorithm is tried, I will be updating the version and also try to provide a summary of accuracy improvement.


## Author
**Ashish Kumar**

[![LinkedIn][1]][2]         [![GitHub][3]][4]      [![Medium][5]][6]

[1]:  Linkedin.png
[2]:  https://www.linkedin.com/in/ashish568/
[3]:  Github.png
[4]:  https://github.com/ashishkr568
[5]:  medium.png
[6]:  https://medium.com/@ashish.568
