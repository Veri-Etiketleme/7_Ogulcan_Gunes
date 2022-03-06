# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 18:07:43 2019

@author: Ganesh
"""

import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import Data_extraction_and_summary as data_extract_file
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def univariateAnalysisNumerical(numerical_data):
    global numerical_columns
    numerical_columns = numerical_data.columns

    #Distplots
    for column in numerical_columns:
        plt.figure()
        sns.distplot(numerical_data[column])
     
    #Min max mode count
    for column in numerical_columns:
        print(numerical_data[column].describe())

def univariateAnalysisCategorical(categorical_data):
    global categorcal_cols
    categorcal_cols =  categorical_data.columns
    
    #Bar plot
    for column in categorcal_cols:
        plt.figure()
        categorical_data[column].value_counts().plot.bar(title = column)

def bivaraiteAnalysisNumandNum(numerical_data):
    global numerical_cols
    numerical_cols = numerical_data.columns
    for col in numerical_cols:
        plt.figure()
        sns.barplot(numerical_data['Attrition'],numerical_data[col])
      
def bivariateAnalysisNumandCat(categorical_data,target_variable):
    global categorical_cols
    categorical_cols = categorical_data.columns
    for col in categorical_cols:
        plt.figure()
        pandas.crosstab(target_variable,categorical_data[col]).plot(kind="bar", stacked=True, figsize=(4,4))         
    
def heatMap(numerical_data):
    matrix = input_file.corr()
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");
    
def removeUnwantedFeatures(input_file):
    #identified columns which can be removed are
    global cols 
    
    #identified uncorrelated columns and intercorrelated columns
    cols =['EmployeeCount','EmployeeNumber','Over18','EducationField','StandardHours','StockOptionLevel','JobLevel','HourlyRate','DailyRate','MonthlyRate','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager' ,'PercentSalaryHike','PerformanceRating']
    input_file.drop(cols, axis=1, inplace=True)
    return input_file
    
def perform_PCA(data):
    #Standardise the data
    std_scaler = StandardScaler()
    data = std_scaler.fit_transform(data)
    #Apply PCA
    pca = PCA(n_components = 1)
    return pandas.DataFrame(pca.fit_transform(data))

def missingValueImputations(data):
    print(data.isnull().sum())

def outlierHandler(data):
    global columns
    columns = data.columns
    
    for col in columns:
        plt.figure()
        data[col].value_counts().plot.box(title=col)        
        
if __name__ == '__main__':
    
    #Define the path for reading the data
    read_path = "D:\Employee Attrition\Employee_Attrition_transformed_data.csv"
    write_path = "D:\Employee Attrition"
    write_path_file_name = "Employee_Attrition_transformed_data_2.csv"
    
    #read the data
    input_file = data_extract_file.readCSV(read_path)
    
    #numerical and categorical data
    numerical_data, categorical_data = data_extract_file.separateCategoricalAndNumerical(input_file)
    
    #Univariate Analysis for Numerical columns
    univariateAnalysisNumerical(numerical_data)
    
    #Univariate Analysis for Categorical columns
    univariateAnalysisCategorical(categorical_data)
    
    #BivariateAnalysis of Numerical & target columns
    bivaraiteAnalysisNumandNum(numerical_data)
    
    #Bivariate Analysis of Target & categorical columns
    bivariateAnalysisNumandCat(categorical_data, numerical_data['Attrition'])
    
    #Finding correlation using heat map
    heatMap(input_file)
    
    #It is found that some columns are intercorrelated 
    #performing PCA for those columns alone
    data_to_perform_pca = input_file[['YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager' ]] 
    data_to_perform_pca_1 = input_file[['PercentSalaryHike','PerformanceRating']]
    input_file['PCA_Years'] = perform_PCA(data_to_perform_pca)
    input_file['PCA_Performance_and_Percent'] = perform_PCA(data_to_perform_pca_1)
    
    #removing unwanted variables
    input_file = removeUnwantedFeatures(input_file)
    
    #Missing values imputation
    missingValueImputations(input_file)
    
    #Outliers identification
    outlierHandler(input_file)
    
    #writing transformed data
    data_extract_file.writeCSV(input_file, write_path,write_path_file_name)
    