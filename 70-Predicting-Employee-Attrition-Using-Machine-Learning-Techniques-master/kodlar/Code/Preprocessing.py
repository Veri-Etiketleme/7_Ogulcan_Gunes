import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load file and randomly shuffle
X=pd.read_excel('/home/ldua/WA_Fn-UseC_-HR-Employee-Attrition.xlsx')
X=shuffle(X, random_state=1)

# Drop columns whose data is the same in all rows
X = X.drop(['Over18','EmployeeCount', 'StandardHours'],1)

# One hot encoding
one_hot_source_features = ['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus']
one_hot_encoded_features = []
one_hot_feature_sources = defaultdict(lambda : "none")
for column in one_hot_source_features:
	one_hot = pd.get_dummies(X[column], prefix=column)
	for i in one_hot:
		one_hot_feature_sources[i] = column
	X = X.drop([column],1)
	X = X.join([one_hot])
	one_hot_encoded_features = one_hot_encoded_features + list(i for i in one_hot)

# Convert Yes/No to 1/0
yes_no_features = ['OverTime']
columns = list(X)
for i in yes_no_features + ['Attrition']:
	X[i] = list(map(lambda x : 1 if x == 'Yes' else 0, X[i]))

# Normalize/standardize columns that aren't already on a 0-1 scale
other_features = [ 'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager' ]
for i in other_features:
	X[i] = (X[i] - np.mean(X[i]))/np.std(X[i])

# Move EmployeeNumber to the first column and Attrition to the last column
attritionindex = columns.index('Attrition')
columns = columns[:attritionindex] + columns[attritionindex+1:] + [columns[attritionindex]]
employeenumberindex = columns.index('EmployeeNumber')
columns = [columns[employeenumberindex]] + columns[:employeenumberindex] + columns[employeenumberindex+1:]
X = X[columns]

# Write the data in different forms

#X.to_csv('All.WithEmployeeNumber.csv', index=False)
X_no_emp=X.drop(['EmployeeNumber'],1)
#X_no_emp.to_csv('All.NoEmployeeNumber.csv', index=False)

(X_train, X_test) = train_test_split(X, test_size=0.25, random_state=42)

#X_train.to_csv('Train.WithEmployeeNumber.csv', index=False)
X_train_no_emp = X_train.drop(['EmployeeNumber'],1)
X_train_no_emp.to_csv('/home/ldua/ML/Train.NoEmployeeNumber.csv', index=False)

#X_test.to_csv('Test.WithEmployeeNumber.csv', index=False)
X_test_no_emp = X_test.drop(['EmployeeNumber'],1)
X_test_no_emp.to_csv('/home/ldua/ML/Test.NoEmployeeNumber.csv', index=False)
'''
Xit = X.copy()

all_features = one_hot_encoded_features + yes_no_features + other_features
for i in range(len(all_features)):
	for j in range(i + 1, len(all_features)):
		if one_hot_feature_sources[i] == "none" or one_hot_feature_sources[i] != one_hot_feature_sources[j]:
			Xit[all_features[i] + ' * ' + all_features[j]] = Xit[all_features[i]] * Xit[all_features[j]]

columns = list(Xit)

# Move EmployeeNumber to the first column and Attrition to the last column
attritionindex = columns.index('Attrition')
columns = columns[:attritionindex] + columns[attritionindex+1:] + [columns[attritionindex]]
employeenumberindex = columns.index('EmployeeNumber')
columns = [columns[employeenumberindex]] + columns[:employeenumberindex] + columns[employeenumberindex+1:]
Xit = Xit[columns]

Xit.to_csv('AllWithInteractions.WithEmployeeNumber.csv', index=False)
Xit_no_emp=Xit.drop(['EmployeeNumber'],1)
Xit_no_emp.to_csv('AllWithInteractions.NoEmployeeNumber.csv', index=False)

(Xit_train, Xit_test) = train_test_split(Xit, test_size=0.25, random_state=42)

Xit_train.to_csv('TrainWithInteractions.WithEmployeeNumber.csv', index=False)
Xit_train_no_emp = Xit_train.drop(['EmployeeNumber'],1)
Xit_train_no_emp.to_csv('TrainWithInteractions.NoEmployeeNumber.csv', index=False)

Xit_test.to_csv('TestWithInteractions.WithEmployeeNumber.csv', index=False)
Xit_test_no_emp = Xit_test.drop(['EmployeeNumber'],1)
Xit_test_no_emp.to_csv('TestWithInteractions.NoEmployeeNumber.csv', index=False)
'''
