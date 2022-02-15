#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Zhenghao Lu'


import pickle
from sklearn import svm
from sklearn.feature_selection import RFE


# selection characteristic, reduce to 500
def selectioning(x, y, step_num):
    model = svm.SVC(kernel='linear')
    rfe = RFE(estimator=model, n_features_to_select=500, step=step_num)
    rfe.fit(x, y)
    with open('RFE/SVC_RFE_500_{}.p'.format(step_num), 'wb') as fp:
        pickle.dump(rfe, fp)


f = open('data_matrix.p', 'rb')
data_matrix = pickle.load(f)
f.close()

X = data_matrix[..., 1:]
Y = data_matrix[..., 0]
# selectioning(X, Y, 5)
# selectioning(X, Y, 10)
selectioning(X, Y, 50)
# selectioning(X, Y, 100)





