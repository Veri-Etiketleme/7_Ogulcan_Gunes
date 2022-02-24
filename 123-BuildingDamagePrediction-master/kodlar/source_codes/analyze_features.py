# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 11:43:08 2018

@author: JEEL
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#name = "train"

# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
#res = pd.read_csv('..\Dataset\\train.csv')

# Preview the first 5 lines of the loaded data 
#print(res.iloc[0:1, 0:14])

def find_all_distinct(name, featurename):
    # Arguments:
    # name - (string) name of the file to be viewed
    # featurename - (string) name of the feature to be analyzed
    
     # read as panda dataset from csv files  
    building_structure = pd.read_csv('..\Dataset\Building_Structure.csv')
    building_ownership = pd.read_csv('..\Dataset\Building_Ownership_Use.csv')
    res = pd.read_csv("..\Dataset\%s.csv"%name)
    
    # merges data based on building id
    building_data = pd.merge(building_structure, building_ownership, on = 'building_id') 
    res = pd.merge(res, building_data, on = "building_id") 
    
    print(np.unique(res[featurename].values))

# counting number of buidings having the praticular features grade wise 
def analyze_column(index):
    # Arguments:
    # index - index of the feature in dataset
    
    # creating a panda dataset
    res = pd.read_csv('..\Dataset\\train.csv')
    
    # creating a freq array
    freq = [0] * 5
    
    # printing column name
    print(res.columns.values[index])
      
    # counting frequency of each damage grade for particular feature
    for i in range(res.shape[0]):
        if res.iloc[i, index] == 1:
            for j in range(5):
                freq[j] = freq[j] + (res.iloc[i, 2] == "Grade " + str(j + 1)) 

    # printing frequencies
    for i in range(5):
        print(str(i) + " " + str(freq[i]))

# Creating a new dictionary with keys in both dictionary and values as sum of values in both dictionary
def concat(dict1, dict2):
    dict = {}
    
    for key in list(dict1.keys()):
        dict[key] = dict1[key] + dict2[key]
    
    return dict

def plot_continuous(x, y, xlabel, ylabel):
    # plotting the points  
    plt.plot(x, y) 
      
    # naming the x axis 
    plt.xlabel(ylabel) 
    # naming the y axis 
    plt.ylabel(xlabel) 
      
    plt.title('Graph of ' + ylabel + ' vs ' + xlabel)
    plt.show() 

# Plots a graph based on an array of dictionary
def plot_dic(cnt, featurename):
    # Arguments:
    # cnt - an array of dictionary for which the stacked bar graph is to be plotted
    
    plt.figure(figsize=(14,5))
    
    bottom1 = cnt[0]
    bottom2 = concat(bottom1, cnt[1])
    bottom3 = concat(bottom2, cnt[2])
    bottom4 = concat(bottom3, cnt[3]);
           
    p0 = plt.bar(range(len(cnt[0])), list(cnt[0].values()))
    p1 = plt.bar(range(len(cnt[0])), list(cnt[1].values()), bottom=list(bottom1.values()))
    p2 = plt.bar(range(len(cnt[0])), list(cnt[2].values()), bottom=list(bottom2.values()))
    p3 = plt.bar(range(len(cnt[0])), list(cnt[3].values()), bottom=list(bottom3.values()))
    p4 = plt.bar(range(len(cnt[0])), list(cnt[4].values()), bottom=list(bottom4.values()))
   
    plt.ylabel('Count')
    plt.title('Count for each damage grade for each distinct value of ' + featurename)
    plt.xticks(range(len(cnt[0])), list(cnt[0].keys()), fontsize = 10, rotation = 30)
    plt.legend((p0[0], p1[0], p2[0], p3[0], p4[0]), ('Grade1', 'Grade2', 'Grade3', 'Grade4', 'Grade5'))
    plt.show(block=False)

# Check frequency for each damage type for each feaure
def analyze_columns(feature_list, feature_type):
    # Arguments:
    # feature_list - (list of string) list of the features to be analyzed
    
    # loading the panda dataset
    data = pd.read_pickle('../panda_objects/data_train.pkl')
    
    # creating an array of empty dictionaries
    cnt = [dict() for x in range(5)]
    
    # initializing dictionaries
    for i in range(5):
        for feature in feature_list:
            cnt[i][feature] = 0            
    
    # maintaining total cnt to see how many data has one of the feature
    total = 0
    
    # maintaining a flag to check if one data can have more than one feature
    flag = 0
    
    # finding the frequencies by iterating through samples
    for i in range(data.shape[0]):
        # maintaining a flag to see if found any feature
        sample_flag = 0
        
        # iterating through features to find the feature of current sample
        for feature in feature_list:
            if data.loc[i, feature] == 1:
                sample_flag = sample_flag + 1
                for grade in range(5):
                    cnt[grade][feature] = cnt[grade][feature] + (data.loc[i, "damage_grade"] == "Grade " + str(grade + 1))   
    
        if sample_flag > 0:
            total = total + 1
        if sample_flag > 1:
            flag = 1
    
    print("Total data:" + str(total))
    
    if flag == 1:
        print("Can have more than one feature at a time")
    else:
        print("Will have one feature at a time")
    
    # printing frequencies
    for feature in feature_list:
        print(feature)
        for grade in range(5):
            print(str(grade) + " " + str(cnt[grade][feature]))
    
    # plot the array of dictionary
    plot_dic(cnt, feature_type) 

# Check frequency for each damage type in each distinct keyword
def analyze_distinct(featurename):
    # Arguments:
    # featurename - (string) name of the feature to be analyzed
    
    # loading the panda dataset
    data = pd.read_pickle('../panda_objects/data_train.pkl')
    
    # finding the distinct values in a column
    distinct_values = np.unique(data[featurename].values)
    
    # creating an array of empty dictionaries
    cnt = [dict() for x in range(5)]
    
    # initializing dictionaries
    for i in range(5):
        for value in distinct_values:
            cnt[i][value] = 0
    
    # finding the frequencies by iterating through samples
    for i in range(data.shape[0]):
        sample_value = data.loc[i, featurename]
        for grade in range(5):
            cnt[grade][sample_value] = cnt[grade][sample_value] + (data.loc[i, "damage_grade"] == "Grade " + str(grade + 1))   
    
    # printing frequencies
    for value in distinct_values:
        print(value)
        for grade in range(5):
            print(str(grade) + " " + str(cnt[grade][value]))
    
    # plot the array of dictionary
    plot_dic(cnt, featurename)    

def analyze_expected(col, grade, colname):
    # Arguments:
    # col - a panda column
    # grade - column of damage grade
    
    sumDict = {}
    cntDict = {}
    avgDict = {}
    
    distinct_values = np.unique(col.values)
    
    for value in distinct_values:
        sumDict[value] = 0
        cntDict[value] = 0
    
    for i in range(col.shape[0]):
        cntDict[col[i]] = cntDict[col[i]] + 1
        sumDict[col[i]] = sumDict[col[i]] + grade[i]
    
    for value in distinct_values:
        avgDict[value] = sumDict[value] / cntDict[value]
        print(str(value) + " " + str(avgDict[value])) 
        
    x = list(avgDict.keys())
    y = list(avgDict.values())
    plt.plot(x, y)
    plt.xlabel(colname)
    plt.ylabel('Expected Damage')
    plt.title('Expected Damage v/s ' + colname)
    plt.show()

def analyze_range(col, grade, step, start, colname):
    sumDict = {}
    cntDict = {}
    avgDict = {}
    
    for i in range(col.shape[0]):
        curBlock = (colname - start) / step
        sumDict[curBlock] = 0
        cntDict[curBlock] = 0
    
    for i in range(col.shape[0]):
        
        cntDict[col[i]] = cntDict[col[i]] + 1
        sumDict[col[i]] = sumDict[col[i]] + grade[i]
    
    for value in list(sumDict.keys()):
        avgDict[value] = sumDict[value] / cntDict[value]
        print(str(value) + " " + str(avgDict[value])) 
        
    x = list(avgDict.keys())
    y = list(avgDict.values())
    plt.plot(x, y)
    plt.xlabel(colname)
    plt.ylabel('Expected Damage')
    plt.title('Expected Damage v/s ' + colname)
    plt.show()

def check_one_hot_encoding(indices):
    # Arguments:
    # indices - indices of the features where one_hot_encoding should exist
    
    # creating a panda dataset
    res = pd.read_csv('..\Dataset\\train.csv')
    
    # printing columnnames
    for index in indices:
        print(res.columns.values[index])
    
    for i in range(100):
        
        # counting number of ones in each sample
        cnt = 0
        for index in indices:
            cnt = cnt + res.iloc[i, index]
        
        # checking if it is one hot encoded
        if cnt > 1:
            print("fail " + str(i) + " " + str(cnt)) 

def delete_ids(res):
    # Arguments :
    # res - (panda dataset)dataset whose certain columns are to be deleted
    
    del res['building_id']
    del res['district_id_x']
    del res['vdcmun_id_x']
    del res['district_id_y']
    del res['vdcmun_id_y']
    del res['ward_id_y']
    
    # returns the dataset after deleting the columns
    return res

def mergeAndCreate(name):
    # Arguments:
    # namr - name of the file(train / test) for which the file is to be created
    
    # read as panda dataset from csv files  
    building_structure = pd.read_csv('..\Dataset\Building_Structure.csv')
    building_ownership = pd.read_csv('..\Dataset\Building_Ownership_Use.csv')
    res = pd.read_csv("..\Dataset\%s.csv"%name)
    
    # merges data based on building id
    building_data = pd.merge(building_structure, building_ownership, on = 'building_id') 
    res = pd.merge(res, building_data, on = "building_id") 
    
    # delete redundant ids
    delete_ids(res)
    
    return res

# Transforms Grade 1 to 1
def transform_output():
    data_train = pd.read_pickle('../panda_objects/data_train.pkl')
    
    for i in range(data_train.shape[0]):
        for grade in range(5):
            if data_train.loc[i, 'damage_grade'] == 'Grade ' + str(grade + 1):
                data_train.at[i, 'damage_grade'] = grade + 1
    return data_train
        
# Processing Starts here
#data_train = mergeAndCreate('train')
#data_test = mergeAndCreate('test')
#data_train = transform_output()
#data_train.to_pickle('../panda_objects/data_train.pkl')
#data_test.to_pickle('../panda_objects/data_test.pkl')
# Processing ends here

#data_train = pd.read_pickle('../panda_objects/data_train.pkl')
#print(len(np.unique(data_train["ward_id_x"].values)))