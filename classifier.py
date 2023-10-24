#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import csv
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb


def set_classifier(name):
    path = os.path.expanduser("W:\\Crowdsourcing")
    path += "\\loggerstation_iwatsuru"
    path += "\\result"
    path += "\\MAE"
    

    if name == "svr":
        path += "\\SVR"
        parameter_file = os.path.join(path, "parameter_svr_sc_10_model.csv")
        with open(parameter_file, "r") as f:
            reader = list(csv.reader(f, lineterminator = "\n"))
            clf = SVR(kernel = 'rbf', gamma = float(reader[1][0]), C = float(reader[1][1]), epsilon = float(reader[1][2]))
    elif name == "rfr":
        clf = RandomForestRegressor(n_jobs = -1)
    elif name == "lgb":
        path += "\\LGB"
        parameter_file = os.path.join(path, "parameter_lgb_sc_10.csv")
        with open(parameter_file, "r") as f:
            reader = list(csv.reader(f, lineterminator = "\n"))
            clf = {
                'objective' : 'regression',
                'metric' : 'l1',
                #'lambda_l1' : float(reader[user][0]),
                #'lambda_l2' : float(reader[user][1]),
                #'num_leaves' : int(reader[user][2]),
                #'feature_fraction' : float(reader[user][3]),
                #'bagging_fraction' : float(reader[user][4]),
                #'bagging_freq' : int(reader[user][5]),
                #'min_child_samples' : int(reader[user][6]),
                }
    else:
        clf = False

    return clf