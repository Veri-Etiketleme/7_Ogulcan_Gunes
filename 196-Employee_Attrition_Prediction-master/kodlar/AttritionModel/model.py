# Import Libraries
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import pydotplus

# Read Data
data = pd.read_csv("Data/data.csv")

# Data Analysis

# print(data.info())                  # Get data info
# print(data.head())                  # Get Data Headers
# print(data.department.unique())     # Get Unique Departments
# print(data.salary.unique())         # Get Salary Data
# print(data.salary)

# Data Transformation

# Step 1:   Tell Model that the salary column is categorical, Encode the cateogries into numericals
data.salary = data.salary.astype('category')
data.salary = data.salary.cat.reorder_categories(['low', 'high', 'medium'])
data.salary = data.salary.cat.codes
# print(data.salary)


# Step 2:   Convert Department Categorical variables into numerical indicators
# A dummy trap is a situation where different dummy variables convey the same information. In this case, if an employee is, say, from the accounting department (i.e. value in the accounting column is 1), then you're certain that s/he is not from any other department (values everywhere else are 0). Thus, you could actually learn about his/her department by looking at all the other departments.
# For that reason, whenever n dummies are created (in our case, 10), only n - 1 (in your case, 9) of them are enough, and the n-th column's information is already included.
# Therefore, you will get rid of the old department column, drop one of the department dummies to avoid dummy trap, and then the two DataFrames.
departments = pd.get_dummies(data.department)
# data = data.join(departments)
# print(departments.head())
departments = departments.drop("accounting", axis=1)  # drop "accounting" column to avoid "dummy trap"
data = data.drop("department", axis=1)  # drop the old column "department" as you don't need it anymore
data = data.join(departments)  # Join the new dataframe "departments" to your employee dataset: done
# print("Data")
# print(data.info())
# print("Department")
# print(departments.info())R
# data.to_csv("data_after_transformation.csv")

# Step 3:   Diagnostic Analysis
employee_count = len(data)
attrition_rate = data.churn.value_counts() / employee_count * 100

# print("Total Employees : " + str(employee_count))
# print("Attrition Rate  : "  + str(attrition_rate))
# print(attrition_rate.head())

labels = 'Stayed', 'Left'
sizes = attrition_rate[0], attrition_rate[1]  # [215, 130]
colors = ['green', "red"]
explode = (0.1, 0)  # explode 1st slice

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.savefig('output/attrition_rate.png')

# Step 4 : Split Training & Test Data
target = data.churn
features = data.drop("churn", axis=1)
target_train, target_test, features_train, features_test = train_test_split(target,features,test_size=0.25,random_state=42)
target_train.to_csv("TrainingData/target_train.csv")
features_train.to_csv("TrainingData/features_train.csv")
features_test.to_csv("TestingData/features_test.csv")
target_test.to_csv("TestingData/target_test.csv")


#Step 5.1 : Run Model (No Depth)
model = DecisionTreeClassifier(random_state=42)
model.fit(features_train,target_train)
model_score_nodepth = model.score(features_test,target_test)*100
print("Model(No Depth) Accuracy: " + str(model_score_nodepth))

#Step 5.2 : Run Model (Max Depth = 5)
model_depth_5 = DecisionTreeClassifier(max_depth=5, random_state=42)
model_depth_5.fit(features_train,target_train)
model_score_depth_5 = model_depth_5.score(features_test,target_test)*100
print("Model(Depth 5) Accuracy: " + str(model_score_depth_5))

#Step 5.3 : Run Model (Min Leaf Nodes = 100)
model_min_leaf = DecisionTreeClassifier(min_samples_leaf=100, random_state=42)
model_min_leaf.fit(features_train,target_train)
model_score_min_leaf = model_min_leaf.score(features_test,target_test)*100
print("Model(Min Leaf Node) Accuracy: " + str(model_score_min_leaf))

# Step 5.4 : Run Model (Depth=5, Min Leaf Nodes = 100)
model_final = DecisionTreeClassifier(max_depth=5, min_samples_leaf=100, class_weight="balanced", random_state=42)
model_final.fit(features_train,target_train)
model_score_final = model_final.score(features_test,target_test)*100
print("Model (Depth=5, Min Leaf Nodes = 100) Accuracy: " + str(model_score_final))

# Step 6.1   : Evaluate Modal : Accuracy Metrics Precision   
prediction = model_final.predict(features_test)
predictionscore = precision_score(target_test,prediction)
print("Accuracy Metrics Precision: " + str(predictionscore))

# Step 6.2   : Evaluate Modal : Accuracy Metric Recall   
prediction = model_final.predict(features_test)
recallscore = recall_score(target_test,prediction)
print("Accuracy Metrics Precision: " + str(recallscore))

# Step 6.3   : Evaluate Modal : ROC/AUC score
prediction = model_final.predict(features_test)
rocaocscore = roc_auc_score(target_test,prediction)
print("ROC/AUC score: " + str(rocaocscore))



#Step 6.1 :  Export Graph     
feature_names = list(features_train)
target_names = list(target_train.unique())
target_names = ['0','1']
#print(feature_names)
#print(target_names)

dot_data = export_graphviz(model_final,out_file=None,
                     feature_names=feature_names,
                     class_names=target_names,
                     filled=True, rounded=True,
                     impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("output/DecisionTreeGraph.pdf")


# Now it's time to Predict
input_data = pd.read_csv("Data/input.csv")
#print(input_data)
output = "Stay"
if (model.predict(input_data) == 1):
        output = "Leave"

print("This Employee is going to " + output)