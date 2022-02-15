#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Zhenghao Lu'


from androguard import misc
import numpy as np
import pickle
import os


# transfer apk file to characteristic matrix
def get_char_matrix(isbenign):
    char_matrix = None
    f = open('dict_perm_api.p', 'rb')
    dict_perm_api = pickle.load(f)
    f.close()
    f = open('list_perm_api.p', 'rb')
    list_perm_api = pickle.load(f)
    f.close()
    if isbenign:
        my_path = 'DataSet/benign'
    else:
        my_path = 'DataSet/malware'
    try:
        for file in os.listdir(my_path):
            the_path = my_path + '/' + file
            the_array = get_character(dict_perm_api, list_perm_api, the_path, isbenign)
            if the_array is None:
                continue
            if char_matrix is None:
                char_matrix = the_array
            else:
                char_matrix = np.vstack((char_matrix, the_array))
    except IOError as e:
        print(e)
    return char_matrix


# extract API and permission from apk file
def get_character(the_dict, the_list, path, isbenign):
    array = None
    try:
        _, _, dx = misc.AnalyzeAPK(path)
        array = np.zeros(len(the_list) + 1, dtype=bool)
        array[0] = isbenign
        for mess, perms in dx.get_permissions(25):
            mess_name = mess.class_name + mess.name
            if mess_name in the_dict.keys():
                array[the_dict[mess_name] + 1] = True
            for perm in perms:
                if perm in the_dict.keys():
                    array[the_dict[perm] + 1] = True
    except BaseException as e:
        print(e)
    return array


# dump training and test characteristic matrix
matrix1 = get_char_matrix(isbenign=True)
matrix2 = get_char_matrix(isbenign=False)
matrix = np.vstack((matrix1, matrix2))
with open('data_matrix.p', 'wb') as fp:
    pickle.dump(matrix, fp)
