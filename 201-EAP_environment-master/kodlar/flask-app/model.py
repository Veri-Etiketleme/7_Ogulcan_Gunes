from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib as matplot
#import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



df = pd.DataFrame.from_csv('15k.csv', index_col=None)

df.rename(index=str,columns={'satisfaction_level': 'satisfaction','last_evaluation': 'evaluation',
                        'number_projects': 'projectCount',
                        'average_monthly_hours': 'averageMonthlyHours',
                        'time_spent_company': 'yearsAtCompany',
                        'work_accident': 'workAccident',
                        'promotion_last_5_years': 'promotion',
                        'sales' : 'department',
                        'left' : 'turnover'
                        },inplace=True)




front = df['turnover']
df.drop(labels=['turnover'], axis=1,inplace = True)
df.insert(0, 'turnover', front)


#print df.department.unique()
df["department"] = df["department"].astype('category').cat.codes
#print df.department.unique()

#print df.salary.unique()
df["salary"] = df["salary"].astype('category').cat.codes
#print df.salary.unique()


# Create train and test splits
target_name = 'turnover'
X = df.drop('turnover', axis=1)


y=df[target_name]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=123, stratify=y)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,confusion_matrix,classification_report,accuracy_score
print X_test.head()

rf = RandomForestClassifier(
    n_estimators=30,
    max_depth=None,
    min_samples_split=10,
    class_weight="balanced"
    #min_weight_fraction_leaf=0.02
    )
rf.fit(X_train, y_train)
print ("\n\n ---Random Forest Model---")
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))
print ("Random Forest AUC = %2.2f" % rf_roc_auc)

rf_accuracy = accuracy_score(y_test, rf.predict(X_test))
print ("Random Forest Accuracy= %2.2f" %rf_accuracy)

tn, fp, fn, tp=confusion_matrix(y_test,rf.predict(X_test)).ravel()
print "True -ve:",tn
print "True +ve:",tp
print "False -ve:",fn
print "False +ve:",fp
print(classification_report(y_test, rf.predict(X_test)))
print rf.predict_proba(X_test)
print confusion_matrix(y_test,rf.predict(X_test))
print len(rf.decision_path(X_test)[1])



from sklearn.externals import joblib

# save model
joblib.dump(rf, 'model.pkl')
