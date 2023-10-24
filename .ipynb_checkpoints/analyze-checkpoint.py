# -*- coding: UTF-8 -*-


import os
import csv
import numpy as np
import classifier
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


#関数read_csv
def read_csv(filename):
    with open(filename) as f:
        reader = list(csv.reader(f))
        reader = reader[1:]
    return reader


#関数analyze（引数questnum:1～297の数字、use_file:ユーザIDが書かれたリスト）
def analyze(questnum, use_file):
    count = 0 #変数countが表すのはリストxの行数
    count_x = [0]

    y = []
    for user in use_file:
        #print(i)
        path = os.path.expanduser("W:\\Crowdsourcing")
        path += "\\loggerstation_iwatsuru"
        path += user
        reader = read_csv(os.path.join(path, "features.csv")) #csvファイルの読み込み
        #print(reader)

        reader = np.array(reader)
        #print(reader)
        reader = np.delete(reader, 0, axis = 1)
        reader = np.delete(reader, 0, axis = 1)
        reader = np.delete(reader, 0, axis = 1)
        reader = np.delete(reader, 0, axis = 1)
        reader = np.delete(reader, 0, axis = 1)
        reader = np.delete(reader, 0, axis = 1)
        #reader_f = np.delete(reader, 29, axis = 1)
        #print(reader_f)

        reader_f = np.array(reader[:, 28]) #解答時間のみの時

        '''#データを標準化
        sc = StandardScaler()
        sc.fit(reader_f) #学習用データで標準化
        reader_f = sc.transform(reader_f)'''

        #use_feature = []
        #reader_f = reader_f[:, use_feature] #reader_fのサイズは、csvファイルの行数

        reader_label = np.array(reader[:, 29]) #csvファイルの29列目のデータ(値は0 or 1)をリストreader_labelに格納する．正誤を表すラベル．
        reader_label = reader_label.astype(np.float64)
        #print(reader_label)

        y_temp = 0
        for t in range(questnum):
            y_temp += reader_label[t] #1問目からquestnum問目までのラベルの値を足し合わせる

        if count == 0: #リストxに1つもデータがないとき
            temp = np.array([]) #リストtempの初期化
            for t in range(questnum):
                temp = np.append(temp, reader_f[t]) #csvファイル内のt問目のデータ(1～29の特徴量の分)をリストtempに追加
            x = temp
            y.append(y_temp / questnum) #yはラベルの値を足し合わせたものを、回答した全問題数で割って、正答率を求めている．
        else: #リストxにデータが既にあるとき(即ち2人目以降)
            temp = np.array([]) #リストの初期化
            for t in range(questnum):
                temp = np.append(temp, reader_f[t])
            x = np.vstack((x, temp)) #リストtempをxに縦方向に結合．この要領でxには値がどんどん足されていく．
            y.append(y_temp / questnum)
        count += 1

        for j in range(len(reader_f) - questnum): #len(reader_f)はcsvファイルの行数
            temp = np.array([])
            for t in range(questnum):
                temp = np.append(temp, reader_f[t + j + 1])
            x = np.vstack((x, temp))
            y_temp -= reader_label[j]
            y_temp += reader_label[j + questnum]
            y.append(y_temp / questnum)
            count += 1

        count_x.append(count)
        #print(count_x)
        #print(x.shape)
        #print(len(y))

    return x, y, count_x


#関数CLF
def CLF(x, y, count_x, num_examinee, num_feature, questnum): #引数はリストx, y, count_xの他に、ユーザの数を表す変数num_examinee，特徴の数を表す(xの列の数)変数num_feature，一度に分析する問題数を表すquestnumである．

    ans = [] #平均絶対誤差の値を格納するためのリストans
    for i in range(num_examinee): #0～9の値でループ
        #classifierを用いて学習モデルを作る
        #clf = SVR(kernel = 'rbf', C = 1.0, epsilon = 0.1, gamma = 'scale')
        clf = classifier.set_classifier("svr")

        #先に全部を訓練データに加え、テストデータにする部分を除外し、訓練データとする
        x_train = x
        y_train = y
        x_train = np.delete(x_train, slice(count_x[i], count_x[i + 1]), 0)
        y_train = np.delete(y_train, slice(count_x[i], count_x[i + 1]), 0)

        #除外した部分をテストデータとする
        x_test = x[count_x[i] : count_x[i + 1]]
        y_test = y[count_x[i] : count_x[i + 1]]

        '''#データを標準化
        sc = StandardScaler()
        sc.fit(x_train) #学習用データで標準化
        x_train_std = sc.transform(x_train)
        x_test_std = sc.transform(x_test)'''

        clf.fit(x_train, y_train) #学習用データを渡してモデルの学習を行う
        y_reg = clf.predict(x_test) #テストデータの全てで予測する

        '''lgb_train = lgb.Dataset(x_train, y_train)
        lgb_test = lgb.Dataset(x_test, y_test, reference = lgb_train)

        gbm = lgb.train(clf, lgb_train)
        preds = gbm.predict(x_test)
        d = mean_absolute_error(y_test, preds)'''

        d = mean_absolute_error(y_test, y_reg)
        #d = np.sqrt(mean_squared_error(y_test, y_reg))
        ans.append(d)

    return ans


#平均正答率による推定
def Average(x, y, count_x, num_examinee, num_feature, questnum):
    ans = []
    for i in range(num_examinee):
        y_train = y
        y_train = np.delete(y_train, slice(count_x[i], count_x[i + 1]), 0)
        y_test = y[count_x[i] : count_x[i + 1]]

        score = 0
        for j in y_train:
            score += j
        ave = score / len(y_train)

        d = 0
        for j in y_test:
            d += abs(ave - j)
            #d += (ave - j) ** 2
        d /= len(y_test)
        #d = np.sqrt(d)
        ans.append(d)

    return ans


def main():
    #use_file = ["\\guest\\", "\\pre01\\", "\\pre02\\", "\\p01\\", "\\p02\\", "\\p03\\", "\\p04\\", "\\p05\\", "\\p06\\", "\\p07\\", "\\p08\\", "\\p09\\", "\\p10\\"]
    use_file = ["\\p01\\", "\\p02\\", "\\p03\\", "\\p04\\", "\\p05\\", "\\p06\\", "\\p07\\", "\\p08\\", "\\p09\\", "\\p10\\"]

    path = os.path.expanduser("W:\\Crowdsourcing")
    path += "\\loggerstation_iwatsuru"
    path += "\\result"
    path += "\\MAE"
    out_path = os.path.join(path, "SVR")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for p in range(120): #問題数
        result_temp = []
        questnum = p + 1
        print(questnum)

        #関数analyzeの処理
        x, y, count_x = analyze(questnum, use_file)

        y = np.array(y) #yをnp.array型に変換
        num_feature = x.shape[1] #リストxの列数を変数num_featureに格納
        #svr.pyの関数SVRの処理
        ans = CLF(x, y, count_x, len(use_file), num_feature, questnum) #ansは平均絶対誤差の値が格納されたリストが返ってくる
        #ans = Average(x, y, count_x, len(use_file), num_feature, questnum) #平均正答率からの推定の時
        result_temp.append(questnum) #一度に分析する数をリストresult_tempに格納

        sum = 0
        for i in range(len(ans)):
            result_temp.append(ans[i]) #リストresult_tempにリストansの各値を追加する
            sum += ans[i]
        result_temp.append(sum / len(ans)) #リストresult_tempの末尾に全員の平均絶対誤差の平均値を格納
        print(result_temp)

        if not os.path.exists(os.path.join(out_path, "final_result_sc_10_model_answer.csv")):
            with open(os.path.join(out_path, 'final_result_sc_10_model_answer.csv'), 'a') as f:
                writer = csv.writer(f, lineterminator = '\n')
                #header = ["ques_num", "guest", "pre01", "pre02", "p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09", "p10", "Ave"]
                header = ["ques_num", "p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09", "p10", "Ave"]
                writer.writerow(header)

        with open(os.path.join(out_path, 'final_result_sc_10_model_answer.csv'), 'a') as f:
            writer = csv.writer(f, lineterminator = '\n')
            writer.writerow(result_temp)


if __name__ == "__main__":
	main()