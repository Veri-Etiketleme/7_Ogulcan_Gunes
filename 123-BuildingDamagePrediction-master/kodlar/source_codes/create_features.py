import numpy as np
import pandas as pd

# Utility functions for creating input features
# ---------------------------------------------

def view_data(name):
    # Arguments:
    # name - (string) name of the file to be viewed
    
    # read as panda dataset from csv files  
    building_structure = pd.read_csv('..\Dataset\Building_Structure.csv')
    building_ownership = pd.read_csv('..\Dataset\Building_Ownership_Use.csv')
    res = pd.read_csv("..\Dataset\%s.csv"%name)
    
    # merges data based on building id
    building_data = pd.merge(building_structure, building_ownership, on = 'building_id') 
    res = pd.merge(res, building_data, on = "building_id") 
    
    res = res.columns.values
    print(res)

def delete_ids(res):
    # Arguments :
    # res - (panda dataset)dataset whose certain columns are to be deleted
    
    del res['building_id']
    #del res['vdcmun_id']
    #del res['district_id']
    del res['district_id_x']
    del res['vdcmun_id_x']
    #del res['ward_id_x']
    del res['district_id_y']
    del res['vdcmun_id_y']
    del res['ward_id_y']
    #del res['has_repair_started']
    
    # returns the dataset after deleting the columns
    return res

def seperate_geotechnical(res):
    # Arguments :
    # res - (panda dataset)dataset to be worked on
    
    # initially setting all geotechnical_risk to 0
    for i in range(res.shape[0]):
        res.at[i, 'has_geotechnical_risk'] = 0
    
    # finding geotechnical_risk
    for i in range(res.shape[0]):
        if i % 10000 == 0:
            print(i)
        for j in range(5, 12):
            if j != 6 and j != 9 and j != 10 and res.iloc[i][j] == 1:
                res.at[i, 'has_geotechnical_risk'] = 1
    
    # deleting rest geotechnical classes
    for i in range(5, 12):
        del res[res.columns.values[5]]
    
    # returns the dataset after seperating geotechincal into effective and not-effective
    return res

def subtract(res, feature1, feature2):
    # Arguments:
    # res - (panda dataset)dataset to be worked on
    # feature1, feature2 - feature 2 will be stored as feature1 + feature2
    
    # initially setting all geotechnical_risk to 0
    for i in range(res.shape[0]):
        res.at[i, feature2] = res.at[i, feature1] - res.at[i, feature2]
        
    # if needed to delete first feature(decreased accuracy)
    # del res[feature1]

def create_numpy_matrix(name, is_labelled):
    # Arguments : 
    # name - (string)name of the file for which numpy matrix is to be calcluated
    # is_labelled - (boolean)whether the file is labelled or not
    
    # read as panda dataset from csv files  
    building_structure = pd.read_csv('..\Dataset\Building_Structure.csv')
    building_ownership = pd.read_csv('..\Dataset\Building_Ownership_Use.csv')
    res = pd.read_csv("..\Dataset\%s.csv"%name)
    
    # merges data based on building id
    building_data = pd.merge(building_structure, building_ownership, on = 'building_id') 
    res = pd.merge(res, building_data, on = "building_id") 
    
    # find number of unique elements in each column
    #unique_elements = res.apply(lambda x : x.nunique(), axis = 0)
    #print(unique_elements)
    
    # seperating geotechnical classes
    # res = seperate_geotechnical(res)
    
    subtract(res, 'height_ft_pre_eq', 'height_ft_post_eq')
    subtract(res, 'count_floors_pre_eq', 'count_floors_post_eq')
    
    # deletes various ids like district_id, building_id e.t.c.
    res = delete_ids(res) 
    
    # converted to a numpy matrix
    res = res.values
    
    # considering column 1 is output shifts output to last as well as randomly shuffles the data
    if is_labelled:
        feature1 = res[:, 0].reshape(res.shape[0], 1)
        output = res[:, 1].reshape(res.shape[0], 1)
        res = np.append(np.append(feature1, res[:, 2 : res.shape[1]], axis = 1), output, axis = 1)
        
        # random shuffle of train data so that the data is well distributed
        np.random.shuffle(res)
        
    # returns a numpy matrix 
    return res 

def separate_output(res):
    # Arguments :
    # res - (numpy matrix)numpy matrix having original data
    
    # vector having output 
    output_labels = np.zeros((res.shape[0], 5))
    
    # creating 5 output classes
    for grade in range(5):
        output_labels[:, grade] = res[:, res.shape[1] - 1] == "Grade " + str(grade + 1)    
    
    # returns two matrices - first one is the feature matrix and second one is the output matrix 
    return res[:, 0 : res.shape[1] - 1], output_labels

def create_column_mapping(res, y):
    # Arguments :
    # res - (numpy matrix)numpy matrix having common form of data
    
    # Assigning number of samples to m 
    m = res.shape[0]
    
    # creating output matrix
    output = np.zeros((m, 1)).reshape(m, 1)
    
    # Creating y values from 0 to 4
    for i in range(m):
        output[i] = np.argmax(y[i, :].reshape(1, 5), axis = 1)
    
    # Column index of featurized data
    ptr = 0
    
    # Dictionary for mapping of column indices from original data to featue matrix
    dic = {} 
    for column_index in range(res.shape[1]):
        print(column_index)
        # Check if column has strings representing different types 
        # Columns with strings and ids are mapped according to type while integer columns are mapped by column indices
        if isinstance(res[0][column_index], str) or column_index == 1:
            # Finding all distinct types and giving each type a column in feature matrix
            types_list = np.unique(np.array(res[:, column_index]))
            
            # Saving in dictionary as 1_Both 3_other e.t.c. i.e columnIndex_type only if concentration greater than 0.2%
            for type_ind in range(len(types_list)):
                types = types_list[type_ind]
                dic.update({str(column_index) + "_" + str(types): ptr})
                ptr = ptr + 1
        elif column_index == 4 or column_index == 5:
            # Finding all distinct types and giving each type a column in feature matrix
            types_list = np.unique(np.array(res[:, column_index]))
            
            # reverse dictionary mapping type to index
            reverse_dic = {}
            for types_ind in range(len(types_list)):
                reverse_dic.update({str(types_list[types_ind]) : types_ind})
            
            # creating a column and total list
            count = [0] * len(types_list)
            tot = [0] * len(types_list)
            
            # updating count and tot
            for example in range(m):
                current_type_ind = reverse_dic[str(res[example][column_index])]
                count[current_type_ind] = count[current_type_ind] + 1
                tot[current_type_ind] = tot[current_type_ind] + output[example]
            
            # creating dictionary
            for types_ind in range(len(types_list)):
                mean = tot[types_ind] / count[types_ind]
                # storing average grades in dictionary
                dic.update({str(column_index) + "_" + str(types_list[types_ind]) : mean[0]})
            
            # finding column mean for those tests having different ids
            column_mean = sum(tot) / m
            dic.update({str(column_index) + "_mean" : column_mean[0]})
        
            dic.update({str(column_index) : ptr})
            
            # updating pointer
            ptr = ptr + 1
        else:
            dic.update({str(column_index): ptr})
            ptr = ptr + 1
    
    print(dic)
    
    #save column mapping dictionary
    np.save('../numpy_objects/column_mapping.npy', dic)
    
    # returns number of columns
    return ptr
    
def transform_data(original_data, number_of_columns):
    # Arguments :
    # original_data - (numpy matrix)numpy matrix storing original data that has to be transformed into a feature matrix
    # number_of_columns - number of columns
    
    # loading the matrix dictionary
    dic = np.load('../numpy_objects/column_mapping.npy').item()
    
    print(dic)
    
    # Setting dimensions of the feature matrix
    m = original_data.shape[0]
    feature_matrix = np.zeros((m, number_of_columns))
    
    # Looping through the data to create the feature matrix
    for row_index in range(m):
        for column_index in range(original_data.shape[1]):
            current_data = original_data[row_index][column_index]
            
            # Checking datas to do respective mapping            
            if isinstance(current_data, str) or column_index == 1:
                feature_matrix[row_index][dic[str(column_index) + "_" + str(current_data)]] = 1
            else: 
                if column_index == 4 or column_index == 5:
                    if str(column_index) + "_" + str(current_data) in dic:
                        feature_matrix[row_index][dic[str(column_index)]] = dic[str(column_index) + "_" + str(current_data)]
                    else:
                        feature_matrix[row_index][dic[str(column_index)]] = dic[str(column_index) + "_mean"]
                else:
                    feature_matrix[row_index][dic[str(column_index)]] = current_data
    
    #returns the generated feature matrix
    return feature_matrix

def generate_mean_and_sd(train_data, test_data):
    # Arguments :
    # train_data - (numpy matrix)numpy matrix having features of training examples
    # test_data - (numpy matrix)numpy matrix having features of test examples
    
    # merging test and train data
    data = np.append(train_data, test_data, axis = 0)
    
    #calculating mean and standard deviation
    sd = np.std(data, axis = 0, keepdims = True)
    mean = np.mean(data, axis = 0, keepdims = True)
    
    #saving mean and standard deviation for further use
    np.save('../numpy_objects/standard_deviation.npy', sd)
    np.save('../numpy_objects/mean.npy', mean)

def normalize(res):
    # Arguments :
    # res - (numpy matrix)matrix to be normalized
    
    # loading mean and sd
    sd = np.load('../numpy_objects/standard_deviation.npy')
    mean = np.load('../numpy_objects/mean.npy')
    
    # normalizing the matrix using broadcasting
    res = (res - mean) / sd
    
    return res

def split_data(train_data, test_data, y_train):
    # Arguments :
    # train_data - (numpy matrix)numpy matrix having features of training examples
    # test_data - (numpy matrix)numpy matrix having features of test examples
    # y_train - (numpy matrix)numpy matrix having output of training examples
    
    # spliting train data into training data and cross validation data 
    training_data = train_data[0 : train_data.shape[0] - 50000, :]
    validation_data = train_data[train_data.shape[0] - 50000 : train_data.shape[0], :]
    y_training = y_train[0 : y_train.shape[0] - 50000, :]    
    y_validation = y_train[y_train.shape[0] - 50000 : y_train.shape[0], :]
    
    # transposing the matrices for easy computations in neural network
    training_data = training_data.T
    validation_data = validation_data.T
    test_data = test_data.T
    y_training = y_training.T
    y_validation = y_validation.T
    
    # saving training, validation and test data
    np.save('../numpy_objects/X_train.npy', training_data)
    np.save('../numpy_objects/X_validation.npy', validation_data)
    np.save('../numpy_objects/X_test.npy', test_data)
    np.save('../numpy_objects/y_train.npy', y_training)
    np.save('../numpy_objects/y_validation.npy', y_validation)

# Data checking functions
# -----------------------

def analyze_output(y):
    # Arguments:
    # y - output
    
    # Assignug number of examples to m
    m = y.shape[1]
    
    # dictionary to save the percentage concentration of each grade for further use
    percentage_concentration = {}    
    
    # counting number of examples with each grade example by example and grade by grade
    for grade in range(5):
        cnt = 0;
        for example in range(m):
            if y[grade][example] == 1:
                cnt = cnt + 1
        print("Grade " + str(grade + 1) + ": " + str(cnt))
        percentage = cnt * 100 / m
        print("Grade " + str(grade + 1) + ": " + str(percentage))
        percentage_concentration["Grade " + str(grade)] = percentage
        
    # saving the concentration percentage dictionary for further use
    np.save("../numpy_objects/percentage_concentration", percentage_concentration)

def reshape_output(y):
    # Arguments :
    # y - (numpy matrix) a 5 x m matrix having output data for binary classification
    
    # assigning number of examples to m
    m = y.shape[1]
    
    # creating a new 1 x m matrix
    res = np.zeros((1, m))
    
    # setting grades from 1 to 5
    for grade in range(5):
        res = res + (grade + 1) * (y[grade, :] == 1)
    
    # returns a 1 x m numpy matrix having output data for regression
    return res
   
# Support snippets for creating training data
# --------------------------------------------

#view_data("train")



#train_data = np.load("../numpy_objects/original_train_data.npy")
#test_data = np.load("../numpy_objects/original_test_data.npy")
#y_train = np.load("../numpy_objects/y_train.npy")

# Start from here
#train_data = create_numpy_matrix("train", True)
#test_data = create_numpy_matrix("test", False)
#
#train_data, y_train = separate_output(train_data)
#
#print(train_data[0, :])
#
#number_of_column = create_column_mapping(train_data, y_train)
#
#train_data = transform_data(train_data, number_of_column)
#test_data = transform_data(test_data, number_of_column)
#
#train_data[np.isnan(train_data)] = 0
#test_data[np.isnan(test_data)] = 0
#y_train[np.isnan(y_train)] = 0
#
#generate_mean_and_sd(train_data, test_data)
#
#train_data = normalize(train_data)
#test_data = normalize(test_data)
#
#split_data(train_data, test_data, y_train)
# End here
    




#np.save("../numpy_objects/original_train_data.npy", train_data)
#np.save("../numpy_objects/original_test_data.npy", test_data)
#np.save("../numpy_objects/y_train.npy", y_train)


        
#y_train = np.load("../numpy_objects/y_train.npy")
#print("Training:")
#analyze_output(y_train)
#
#y_validation = np.load("../numpy_objects/y_validation.npy")
#print("Validation:")
#analyze_output(y_validation)

#y_train_regression = reshape_output(y_train)
#y_validation_regression = reshape_output(y_validation)
#
#np.save("../numpy_objects/y_train_regression.npy", y_train_regression)
#np.save("../numpy_objects/y_validation_regression.npy", y_validation_regression)