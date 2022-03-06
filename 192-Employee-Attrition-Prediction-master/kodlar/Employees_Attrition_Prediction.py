#import libraries
import pandas as pd
import numpy as np

AttritionData = pd.ExcelFile("Company_X_Employees_Data.xlsx")

#now parse the sheets as dataframes into a variable
Employees_Left = AttritionData.parse("Employees who have left")
Employees_Existing = AttritionData.parse("Existing employees")

#merge both dataframes/sheets but before that insert a new column (Attrition) in each dataframe

#Get the length of the rows in each dataframe
len1 = Employees_Left.shape[0]
len2 = Employees_Existing.shape[0]

#create a dummy random array equal in length to each dataframe
dummyArray1 = np.arange(len1)
dummyArray2 = np.arange(len2)

#Re-assign the values of the dummy array respectively, such that, say Attrition in Employees who left dataframe is 1
# while in the other dataframe is 0
for x in dummyArray1:
    dummyArray1[x] = 1

for n in dummyArray2:
    dummyArray2[n] = 0

#insert the dummy arrays as new columns in the dataframes
Employees_Left["Attrition"] = dummyArray1
Employees_Existing["Attrition"] = dummyArray2

#merge existing employees to those who left using concat, use ignore_index=True so as to reassign the index after merge
employees = pd.concat([Employees_Left, Employees_Existing], ignore_index=True)

#enable all columns to show during print, instead of being truncated
pd.set_option('max_columns', None)

#Convert Categorical values to Numeric Values
employees = pd.get_dummies(employees)

#Separating the dependent and independent values
X = employees.drop(['Attrition'], axis=1)
Y = employees['Attrition']

#Scaling the data values to standardize the range of independent variables (Feature Scaling)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)

#Function to Train and Test Any Model/Algorithm
def my_train_test_model(X_train, Y_train, X_test, Model):
    model.fit(X_train, Y_train) #Training the model, by fitting the model to the Training set
    Y_pred = model.predict(X_test) # Predicting Test set results

    # Test the Model
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test, Y_pred)
    print('The confusion matrix of the', Model, 'Model is:')
    print(cm)

    # Accuracy Score
    from sklearn.metrics import accuracy_score
    accuracy = round(accuracy_score(Y_test, Y_pred) * 100, 2)
    print('Accuracy of the', Model, 'Model is', str(accuracy)+'%')
    print(" ")

#Calling the function defined earlier to make the prediction, using different Algorithms

#Prediction using SVM Algorithm or model
from sklearn.svm import SVC  #Import Model
ModelName = "SVC"
model = SVC() #instantiating the model and assigning to a variable
my_train_test_model(X_train, Y_train, X_test, ModelName) #calls the function define

#Prediction using Decision Tree model
from sklearn.tree import DecisionTreeClassifier
ModelName = "DecisionTreeClassifier"
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
my_train_test_model(X_train, Y_train, X_test, ModelName) #calls the function define

#Prediction using Random Forest Model
from sklearn.ensemble import RandomForestClassifier
ModelName = "RandomForestClassifier"
model = RandomForestClassifier()
my_train_test_model(X_train, Y_train, X_test, ModelName)