import pandas as pd
import numpy as np

training_set = pd.read_csv('/home/ldua/ML/Train.NoEmployeeNumber.csv')
testing_set = pd.read_csv('/home/ldua/ML/Test.NoEmployeeNumber.csv')

training_set.Attrition.replace([0, 1], ['yes', 'no'], inplace=True)
testing_set.Attrition.replace([0, 1], ['yes', 'no'], inplace=True)

mean = training_set.groupby('Attrition').mean()
var = training_set.groupby('Attrition').var()

# test_set_mod = df.drop(['DraftYear', 'GP_greater_than_0', 'country_group', 'Position'], axis=1)
# del testing_set['DraftYear']

test_set_mod = testing_set.drop(['Attrition'], axis=1)

count_training = training_set.groupby('Attrition').size()
count_training_yes = count_training['yes']
count_training_no = count_training['no']
count_total = count_training_no + count_training_yes


count_test = testing_set.groupby('Attrition').size()
count_test_yes = count_test['yes']
count_test_no = count_test['no']


count_testing = len(testing_set)
p_yes = float(count_training_yes/count_total)
p_no = float(count_training_no/count_total)
inc = 0
pi = np.pi

x1 = float(count_training_yes/(count_training_yes - 1))
x2 = float(count_training_no/(count_training_no - 1))
var2 = var.mul([x1, x2], axis=0)

for i in (testing_set.index):

    p_yy = 0
    p_nn = 0

    for j in test_set_mod:
        power_yes = (np.square(testing_set.at[i, j] - mean.at['yes', j])) / (-2 * var2.at['yes', j])
        p_x_given_yes = np.log((np.exp(power_yes)) / (np.sqrt(2 * pi * (var2.at['yes', j]))))
        p_yy = p_yy + p_x_given_yes

        power_no = (np.square(testing_set.at[i, j] - mean.at['no', j])) / (-2 * var2.at['no', j])
        p_x_given_no = np.log((np.exp(power_no)) / (np.sqrt(2 * pi * (var2.at['no', j]))))
        p_nn = p_nn + p_x_given_no

    row_p_yes = p_yy + np.log(p_yes)
    row_p_no = p_nn + np.log(p_no)

    if (testing_set.at[i, 'Attrition'] == 'yes'):
        if (row_p_yes > row_p_no):
            inc = inc + 1

    if (testing_set.at[i, 'Attrition'] == 'no'):
        if (row_p_yes < row_p_no):
            inc = inc + 1

accuracy = (inc/count_testing)*100
print("----------------------------")
print('Accuracy =', accuracy)
print("----------------------------")
