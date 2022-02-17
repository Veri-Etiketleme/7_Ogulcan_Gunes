import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# data = pd.read_csv('/home/ldua/AttritionNoEmployeeNumber.csv')
# Y = data['Attrition']
# X = data.drop('Attrition', 1)


# Creating training data, training target, test data, test traget
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Creating training data, training target, test data, test traget
# def create_test_train(X, Y):
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42) #-----changed-----
#     return X_train, X_test, Y_train, Y_test

# def interaction(X):
#     interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
#     X_inter = interaction.fit_transform(X)
#     return X_inter

# k-fold cross-validation with k = 5
def cross_validation(k, X_train, Y_train):

    s = X_train.shape[0]
    x=(round(s/k)) #1
    a=0

    max = .5
    for i in range(k):  #1102

        a = x*i    # 0, 63, 126
        b = a+x

        x_test = X_train[a:b]
        y_test = Y_train[a:b]

        x_train = np.concatenate((X_train[:a], X_train[b:]))
        y_train = np.concatenate((Y_train[:a], Y_train[b:]))

        #print(train.shape)
        #print(test.shape)
        #print("----")

        model = GaussianNB()

        sc_x = StandardScaler()
        x_train_s = sc_x.fit_transform(x_train)
        x_test_s = sc_x.fit_transform(x_test)

        model.fit(x_train_s, y_train)

        y_pred = model.predict(x_test_s)

        # Confusion Matrix
        ##cm = confusion_matrix(y_test, y_pred)
        ##accuracy = (cm[0][0] + cm[1][1]) / len(y_test)

        accuracy = model.score(x_test_s, y_test)


        print("\n", accuracy)
        #print(cm)

        # Choosing the model that lead to best accuracy
        if max < accuracy:
            max = accuracy
            best_model = model

    return max, best_model


train = pd.read_csv('/home/ldua/ML/Train.NoEmployeeNumber.csv')
Y_train = train['Attrition']
X_train = train.drop('Attrition', 1)

test = pd.read_csv('/home/ldua/ML/Test.NoEmployeeNumber.csv')
Y_test = test['Attrition']
X_test = test.drop('Attrition', 1)

k = 5
max, best_model = cross_validation(k, X_train, Y_train)

print("\n---------------- PREDICTING FOR TEST SET WITH BEST MODEL ----------------")
print("Best Accuracy during cross validation was: ", max)

sc_x = StandardScaler()
X_test = sc_x.fit_transform(X_test)

y_pred = best_model.predict(X_test)

cm = confusion_matrix(Y_test, y_pred)
accuracy = (cm[0][0] + cm[1][1]) / len(Y_test)

#np.savetxt('/home/ldua/naive/',y_pred)

print(accuracy)
print(cm)