#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import classifier
from sklearn.preprocessing import StandardScaler


class FeatureSelection:

    def __init__(self, user):
        self.user = user
        self.size = 1 #1 or 5 or 10
        path = os.path.expanduser("G:\マイドライブ\Crowdsourcing")
        path += "\loggerstation_iwatsuru"

        #要変更
        path += self.user
        self.path = path
        #print(self.path)

        #分類:svm, rf, nb, gb
        #回帰:svr, rfr
        self.clf_name = "svr"

        self.read()

    def read(self):
        df = pd.read_csv(os.path.join(self.path, "features.csv"))
        data = df.values
        #print(df)

        self.n_class = 3
        self.id = list(df["#id"].values.tolist())
        self.columns = list(df.columns)

        #class label
        self.y = data[:, 5]

        #features
        self.x = np.delete(data, 0, axis = 1)
        self.x = np.delete(self.x, 0, axis = 1)
        self.x = np.delete(self.x, 0, axis = 1)
        self.x = np.delete(self.x, 0, axis = 1)
        self.x = np.delete(self.x, 0, axis = 1)
        self.x = np.delete(self.x, 0, axis = 1)
        self.x = np.delete(self.x, 29, axis = 1)

        self.feature_selection()


    def feature_selection(self):
        clf = classifier.set_classifier(self.clf_name)

        #for j in range(col):
        #    ave = np.average(x[:, j])
        #    std = np.std(x[:, j])
        #    if std != 0:
        #        x[:, j] = (self.x[:, j] - ave) / std

        ave_score_about = []

        row = self.x.shape[0]
        num = range(row)
        self.predict_y = np.array([])
        for i in range(0, row, self.size):
            idx = num[i : i + self.size if (i + self.size) < row else row]
            x_test = np.array(self.x[idx], dtype = 'float')
            y_test = np.array(self.y[idx], dtype = 'float')
            x_train = np.array(np.delete(self.x, idx, 0), dtype = 'float')
            y_train = np.array(np.delete(self.y, idx, 0), dtype = 'float')

            #データを標準化
            sc = StandardScaler()
            sc.fit(x_train) #学習用データで標準化
            x_train_std = sc.transform(x_train)
            x_test_std = sc.transform(x_test)

            #fs = sfs(clf, k_features = 5, forward = True, floating = False, verbose = 2, n_jobs = -1, scoring = 'r2', cv = 5)
            #fs = fs.fit(x_train_std, y_train)
            #indices = list(fs.k_feature_idx_)

            #x_train_std = x_train_std[:, indices]
            #x_test_std = x_test_std[:, indices]
            #x_train_fs = fs.transform(x_train_std)
            #x_test_fs = fs.transform(x_test_std)

            clf.fit(x_train_std, y_train)
            #回帰
            d = clf.predict(x_test_std)
            #評価
            self.predict_y = np.append(self.predict_y, d)

        score_about = []
        for i in range(row):
            if ((self.predict_y[i] < 4) & (self.y[i] < 4) | (4 <= self.predict_y[i]) & (4 <= self.y[i])):
                score_about.append(1)
            else:
                score_about.append(0)
        #print(score_about)

        ave_score_about.append(np.average(score_about))

        pred = pd.DataFrame()
        pred["id"] = self.id
        pred["label"] = self.y
        pred["pred_label"] = self.predict_y
        pred["score_about"] = score_about

        if not os.path.exists(os.path.join(self.path, "Pred")):
            os.makedirs(os.path.join(self.path, "Pred"))
        pred_path = os.path.join(self.path, "Pred")
        pred.to_csv(os.path.join(pred_path, "pred_" + self.clf_name + ".csv"))
        print(ave_score_about)


FeatureSelection("/guest")