## AtPred

 Employee attrition analytics is specifically focused on why employees voluntarily leave their jobs, and identifying contributing factors. AtPred is a utility for predicting if a given employee is at risk of leaving their employer. 

### Run

1. Clone the repo
2. Start an Anaconda3 Prompt and activate the included environment
```
conda env create -f environment.yml
```
3. Run AtPred.py and specify the employee data file
```
python AtPred.py example_data.csv
```



Running the main application will:

1. Convert the provided employee data to numerical features used for our support vector machine (SVM) model
2. Use our SVM model to predict whether the provided employees are at risk of quitting.
3. Display a binary array of results indicating if the corresponding employees are either at risk (1) or not at risk (0) 



Employee data must be formatted as a csv value, having all of the column data present in the checked in file example_data.csv. Column headers are required.

### Machine Learning Model

Our model is based on a support vector machine that was trained against IBM HR employee data, available at :

https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset#WA_Fn-UseC_-HR-Employee-Attrition.csv

