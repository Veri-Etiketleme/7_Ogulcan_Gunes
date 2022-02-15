#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Zhenghao Lu'


import pickle
import numpy as np
import sys
import time
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import svm
import matplotlib.pyplot as plt


# cluster algorithm
def clustering(training):
    # calculate the number of clusters
    _, sigma, _ = np.linalg.svd(training)
    sig2 = sigma ** 2
    energy = sum(sig2)
    cluster_size = 0
    for i in range(500):
        if sum(sig2[:i]) > 0.9 * energy:
            cluster_size = i
            break
    cluster_size -= 1
    # cluster algorithm
    kmean = KMeans(n_clusters=cluster_size)
    kmean.fit(training)
    return kmean, cluster_size


# train classifiers
def save_classifier(classifier_no, cluster_length, clu_index, x, y):
    clf_l = []
    for k in range(0, cluster_length):
        clusteratindex = np.where(clu_index == k)[0]
        xn = x[clusteratindex, ...]
        yn = y[clusteratindex, ...]
        if yn[yn == True].shape[0] == yn.shape[0]:
            clf_l.append(True)
        elif yn[yn == False].shape[0] == yn.shape[0]:
            clf_l.append(False)
        else:
            if classifier_no == 1:
                clf = KNeighborsClassifier(algorithm='auto')
            elif classifier_no == 2:
                clf = svm.SVC()
            elif classifier_no == 3:
                clf = RandomForestClassifier()
            else:
                clf = DecisionTreeClassifier()
            clf.fit(xn, yn)
            clf_l.append(clf)
    return clf_l


# find the specific cluster for the sample
def find_cluster(app, cluster):
    min_distance = sys.maxsize
    min_index = 0
    for k, cluster_center in enumerate(cluster.cluster_centers_):
        my_sum = 0
        for m, point in enumerate(cluster_center):
            my_sum += (app[m] - point) ** 2
        if my_sum < min_distance:
            min_distance = my_sum
            min_index = k
    return min_index


# calculate TP, FP, TN, FN
def get_results(x, y, cluster, classifier):
    tp, fp, tn, fn = 0, 0, 0, 0
    for k, application in enumerate(x):
        cluster_no = find_cluster(application, cluster)
        the_app = application.reshape(1, -1)
        clf = classifier[cluster_no]
        if isinstance(clf, bool):
            ans = clf
        else:
            ans = clf.predict(the_app)[0]
        if ans:
            if y[k]:
                tn += 1
            else:
                fn += 1
        else:
            if y[k]:
                fp += 1
            else:
                tp += 1
    return tp, fp, tn, fn


# calculate accuracy, precision and recall
def get_result(x, y, cluster, classifier_list):
    result = []
    for classifier in classifier_list:
        # start_time = time.time()
        tp, fp, tn, fn = get_results(x, y, cluster, classifier)
        # end_time = time.time()
        # print(end_time - start_time)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        result.append([accuracy, precision, recall])
    return result


# plot ROC curve
def plot_roc(class_cluster, cluster_no, clf_name):
    the_clf = class_cluster[cluster_no]
    if clf_name == 'SVM':
        y_score = the_clf.decision_function(test_x)
    else:
        y_score = the_clf.predict_proba(test_x)[..., 1]
    fpr, tpr, threshold = roc_curve(test_label, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label='%s (area = %0.2f)' % (clf_name, roc_auc))


# main code
f = open('data_matrix.p', 'rb')
data_matrix = pickle.load(f)
f.close()
f = open('SVC_RFE_500_50.p', 'rb')
rfe = pickle.load(f)
f.close()
# data matrix with 500 characteristics
ranking = rfe.ranking_
index = np.where(ranking == 1)[0]
index = np.concatenate(([0], np.add(index, 1)))
data_matrix = data_matrix[..., index]
# shuffle
np.random.shuffle(data_matrix)
test_size = data_matrix.shape[0] // 5
split_data = []
for i in range(0, 5):
    if i == 4:
        one_five = data_matrix[(i * test_size):]
    else:
        one_five = data_matrix[(i * test_size):((i + 1) * test_size)]
    split_data.append(one_five)
my_result = []
for i in range(0, 5):
    print(str(i + 1) + ' in 5')
    training_data = None
    test_data = split_data[i]
    for j in range(0, 5):
        if j != i:
            if training_data is None:
                training_data = split_data[j]
            else:
                training_data = np.vstack((training_data, split_data[j]))
    # cluster algorithm
    training_label = training_data[..., 0]
    training_x = training_data[..., 1:]
    test_label = test_data[..., 0]
    test_x = test_data[..., 1:]
    my_cluster, cluster_num = clustering(training_x)
    cluster_index = my_cluster.predict(training_x)
    cls_list = []
    # knn == 1
    cls_list.append(save_classifier(1, cluster_num, cluster_index, training_x, training_label))
    # svm == 2
    cls_list.append(save_classifier(2, cluster_num, cluster_index, training_x, training_label))
    # random forest == 3
    cls_list.append(save_classifier(3, cluster_num, cluster_index, training_x, training_label))
    # decision tree == 4
    cls_list.append(save_classifier(4, cluster_num, cluster_index, training_x, training_label))
    # get result
    my_result.append(get_result(test_x, test_label, my_cluster, cls_list))


# print rsults and save them in csv
f = open('result.csv', 'w')
f.write('Classification Algorithm,Accuracy,Precision,Recall\n')
my_result = np.asarray(my_result, dtype=float)
print('\n')
for i in range(0, 4):
    if i == 0:
        str1 = 'KNN'
    elif i == 1:
        str1 = 'SVM'
    elif i == 2:
        str1 = 'Random Forest'
    else:
        str1 = 'Decision Tree'
    f.write(str1 + ',')
    print(str1)
    for j in range(0, 3):
        mean = np.mean(my_result[..., i, j])
        if j == 0:
            str2 = 'accuracy'
            f.write(str(mean) + ',')
        elif j == 1:
            str2 = 'precision'
            f.write(str(mean) + ',')
        else:
            str2 = 'recall'
            f.write(str(mean))
        print(str2 + ': ' + str(mean))
    f.write('\n')
    print('\n')
f.close()












