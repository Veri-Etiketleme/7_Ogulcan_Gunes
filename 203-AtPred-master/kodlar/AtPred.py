#File: AtPred.py
#Author: Austin Spadaro
#Description: A command line utility to predict attrition. 
#Notes: This program was developed against the Python runtime version 3.9.5

import sys
import os.path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

#First do some input validation.
if len(sys.argv) < 2:
    print('AtPred.py <employee data file path>')
    quit()

if not os.path.exists(sys.argv[1]):
    print('The specified employee data file does not exist.')
    quit()
    
if not os.path.exists('Models\AtPred_Model.sav'):
    print('The AtPred model does not exist.')
    quit()

#Convert the employee data to numerical features.
    
#Read in our data as a csv file.
dataframe = pd.read_csv(sys.argv[1])

#First let's convert our binary fields.
dataframe = dataframe.drop(['EmployeeCount'], axis=1)
dataframe = dataframe.drop(['EmployeeNumber'], axis=1)
dataframe = dataframe.drop(['StandardHours'], axis=1)

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

#When we encode the provided data, not all possible values will be present. We need to add anything missing so the feature set matches what was used to train the model.
encoded_columns = "BusinessTravel_Non-Travel,BusinessTravel_Travel_Frequently,BusinessTravel_Travel_Rarely,Department_Human Resources,Department_Research & Development,Department_Sales,EducationField_Human Resources,EducationField_Life Sciences,EducationField_Marketing,EducationField_Medical,EducationField_Other,EducationField_Technical Degree,JobRole_Healthcare Representative,JobRole_Human Resources,JobRole_Laboratory Technician,JobRole_Manager,JobRole_Manufacturing Director,JobRole_Research Director,JobRole_Research Scientist,JobRole_Sales Executive,JobRole_Sales Representative,MaritalStatus_Divorced,MaritalStatus_Married,MaritalStatus_Single".split(",")
for col in encoded_columns:
    if not col in dataframe.columns:
        dataframe[col] = 0

#Order our columns to match the order what was used in training.
dataframe = dataframe[['Age','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction','Gender','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome','MonthlyRate','NumCompaniesWorked','Over18','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','BusinessTravel_Non-Travel','BusinessTravel_Travel_Frequently','BusinessTravel_Travel_Rarely','Department_Human Resources','Department_Research & Development','Department_Sales','EducationField_Human Resources','EducationField_Life Sciences','EducationField_Marketing','EducationField_Medical','EducationField_Other','EducationField_Technical Degree','JobRole_Healthcare Representative','JobRole_Human Resources','JobRole_Laboratory Technician','JobRole_Manager','JobRole_Manufacturing Director','JobRole_Research Director','JobRole_Research Scientist','JobRole_Sales Executive','JobRole_Sales Representative','MaritalStatus_Divorced','MaritalStatus_Married','MaritalStatus_Single']]

#A module for scaling our empoyee data.
scaler = StandardScaler()

pred_data = scaler.fit_transform(dataframe)

#load model
model = pickle.load(open('Models\AtPred_Model.sav', 'rb'))

#Make a prediction.
result = model.predict(pred_data)

print(result)

