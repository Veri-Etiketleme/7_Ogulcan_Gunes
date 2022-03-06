# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 10:16:32 2019

@author: Ganesh
"""

############### EMPLOYEE CHURN PREDICTION ################

############### IBM SAMPLE DATASET ##################

#import packages
import pandas
import os

#Function for reading the data 
def readCSV(path):  
    global input_data
    #Check whether the file exists or not
    exists = os.path.isfile(path)
    if exists:
        input_data = pandas.read_csv(path)
    else:
        print("File Does not exist")
    return input_data

#Function to separate numerical and categorical data
def separateCategoricalAndNumerical(input_data):   
    global numerical_data, categorical_data,numerical_cols,categorical_cols
    numerical_cols = ['Age', 'Attrition', 'DailyRate', 'DistanceFromHome', 'Education',
       'EmployeeCount', 'EmployeeNumber', 'EnvironmentSatisfaction',
       'HourlyRate', 'JobInvolvement', 'JobLevel',
       'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
       'PercentSalaryHike',
       'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear',  'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']
    categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole',
       'MaritalStatus', 'Over18', 'OverTime','JobSatisfaction','PerformanceRating','RelationshipSatisfaction','WorkLifeBalance']
    numerical_data = input_data[numerical_cols]
    categorical_data = input_data[categorical_cols]
    return numerical_data,categorical_data

#Function to display the summary of the data
def data_summary(input_data):
    global columns_list 
    columns_list = input_data.columns
    for column in columns_list:
        print(input_data[column].describe())

def writeCSV(input_data, write_path,write_path_file_name):
    input_data.to_csv(write_path + "\\" + write_path_file_name)

if __name__ == '__main__':

    #Define the path for reading the data
    read_path = "D:\Employee Attrition\Employee_Attrition.csv"
    write_path = "D:\Employee Attrition"
    write_path_file_name = "Employee_Attrition_transformed_data.csv"
    
    #function to read the datas
    input_data = readCSV(read_path)
    
    #function to separate numerical and categorical variables
    numerical_data, categorical_data = separateCategoricalAndNumerical(input_data)
    
    #print data summary
    data_summary(input_data)
    
    #changing the target variable to numericals
    Attrition_map = {'Yes':1,'No':0}
    input_data['Attrition'] = input_data['Attrition'].map(Attrition_map)
    target_variable = pandas.DataFrame(input_data['Attrition'])
    
    #writing a transformed data
    writeCSV(input_data, write_path,write_path_file_name)
