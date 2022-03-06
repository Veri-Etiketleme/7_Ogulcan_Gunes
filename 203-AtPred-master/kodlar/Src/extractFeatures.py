#File: extractFeatures.py
#Author: Austin Spadaro
#Description: A script for converting our employee attrition data to numerical features.
#Notes: This program was developed against the Python runtime version 3.9.5
import pandas as pd

#Read in our data as a csv file.
dataframe = pd.read_csv("Data\\employee_data.csv")

#Uncomment this line to view a summary of the file data before we modify it.
#print(dataframe.info())

#First let's convert our binary fields.

#Attrition
attrition_values = {'Yes' : 1, 'No' : 0}
dataframe['Attrition'] = dataframe['Attrition'].map(attrition_values)

#Gender
gender_values = {'Female' : 1, 'Male' : 0}
dataframe['Gender'] = dataframe['Gender'].map(gender_values)

#Over18
over18_values = {'Y' : 1, 'N' : 0}
dataframe['Over18'] = dataframe['Over18'].map(over18_values)

#OverTime
overTime_values = {'Yes' : 1, 'No' : 0}
dataframe['OverTime'] = dataframe['OverTime'].map(overTime_values)

#Now we need to convert our categorical fields using one hot encoding.

#BusinessTravel, Department, EducationField, JobRole, and MaritalStatus
dataframe = pd.get_dummies(data=dataframe, columns=['BusinessTravel', 'Department','EducationField','JobRole','MaritalStatus'])

#Output our feature vectors to a new csv file.
dataframe.to_csv('Features\\features.csv',index=False)
