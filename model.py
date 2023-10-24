# -*- coding: UTF-8 -*-


import os
import csv
import numpy as np
import classifier
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
#import optuna.integration.lightgbm as lgb_o
import optuna
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score
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
        reader_f = np.delete(reader, 29, axis = 1)
        #print(reader_f)

        #データを標準化
        sc = StandardScaler()
        sc.fit(reader_f) #学習用データで標準化
        reader_f = sc.transform(reader_f)

        #reader_f = np.array(reader[:, 28]) #解答時間のみの時

        #use_feature = [1, 2, 23, 24]
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


#関数tuning
def tuning(x, y, count_x, num_examinee, num_feature, questnum): #引数はリストx, y, count_xの他に、ユーザの数を表す変数num_examinee，特徴の数を表す(xの列の数)変数num_feature，一度に分析する問題数を表すquestnumである．
    #classifierを用いて学習モデルを作る
    #clf = SVR(kernel = 'rbf', C = 1, epsilon = 0.1, gamma = 'auto')
    #clf = classifier.set_classifier("svr")

    '''ans = [] #平均絶対誤差の値を格納するためのリストans
    for i in range(num_examinee): #0～9の値でループ
        #先に全部を訓練データに加え、テストデータにする部分を除外し、訓練データとする
        x_train = x
        y_train = y
        x_train = np.delete(x_train, slice(count_x[i], count_x[i + 1]), 0)
        y_train = np.delete(y_train, slice(count_x[i], count_x[i + 1]), 0)

        #除外した部分をテストデータとする
        x_test = x[count_x[i] : count_x[i + 1]]
        y_test = y[count_x[i] : count_x[i + 1]]

        #データを標準化
        sc = StandardScaler()
        sc.fit(x_train) #学習用データで標準化
        x_train_std = sc.transform(x_train)
        x_test_std = sc.transform(x_test)'''

    group = []
    count = 0
    for j in range(num_examinee):
        for k in range(count_x[j], count_x[j + 1]):
            group.append(count)
        count += 1
    #print(group)
    logo = LeaveOneGroupOut()
    #cv_gen = logo.split(x_train_std, y_train, groups = group)
    #cv = list(cv_gen)

    '''params = {
        'task' : 'train', #タスクを訓練に設定
        'boosting_type' : 'gbdt', #GBDTを指定
        'objective' : 'regression', #回帰を指定
        'metric' : 'l1', #回帰の損失（誤差）
        'learning_rate' : 0.1, #学習率
        'seed' : 0 #シード値
        }

    best_params, history = {}, []
    cv_result_opt = []

    for i, (tr_idx, va_idx) in enumerate(logo.split(x_train, y_train, groups = group)):
        #trainとvalidに分ける
        tra_x, val_x = x_train[tr_idx], x_train[va_idx]
        tra_y, val_y = y_train[tr_idx], y_train[va_idx]

        lgb_tra = lgb.Dataset(tra_x, tra_y)
        lgb_val = lgb.Dataset(val_x, val_y, reference = lgb_tra)

        lgb_results = {} #学習の履歴を入れる入物

        model = lgb.train(
            params, #ハイパーパラメータをセット
            lgb_tra, #訓練データを訓練用にセット
            valid_sets=[lgb_tra, lgb_val], #訓練データとテストデータをセット
            valid_names=['Train', 'Valid'], #データセットの名前をそれぞれ設定
            num_boost_round = 100, #計算回数
            early_stopping_rounds = 50, #アーリーストッピング設定
            evals_result = lgb_results,
            verbose_eval = -1, #ログを最後の1つだけ表示
            )
        best_params = model.params

        #推論
        y_pred = model.predict(val_x, num_iteration = model.best_iteration)

        #評価
        d = mean_absolute_error(val_y, y_pred)
        cv_result_opt.append(d)

    print("RMSE:", cv_result_opt)
    print("RMSE:", np.mean(cv_result_opt))'''

    #目的関数
    def objective(trial):
        #kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid', 'precomputed'])
        gamma = trial.suggest_float('gamma', 1e-5, 1e5, log = True)
        C = trial.suggest_float('C', 1e-5, 1e5, log = True)
        epsilon = trial.suggest_float('epsilon', 1e-5, 1e5, log = True)

        #regr = SVR(kernel = kernel, gamma = gamma, C = C, epsilon = epsilon)
        regr = SVR(kernel = 'rbf', gamma = gamma, C = C, epsilon = epsilon)

        '''param = {
            'objective' : 'regression',
            'metric' : 'l1',
            'lambda_l1' : trial.suggest_float('lambda_l1', 1e-8, 10.0, log = True),
            'lambda_l2' : trial.suggest_float('lambda_l2', 1e-8, 10.0, log = True),
            'num_leaves' : trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction' : trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction' : trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq' : trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples' : trial.suggest_int('min_child_samples', 5, 100),
            }'''

        ab = []
        for i, (tr_idx, va_idx) in enumerate(logo.split(x, y, groups = group)):
            #trainとvalidに分ける
            tra_x, val_x = x[tr_idx], x[va_idx]
            tra_y, val_y = y[tr_idx], y[va_idx]

            '''lgb_tra = lgb.Dataset(tra_x, tra_y)
            lgb_val = lgb.Dataset(val_x, val_y, reference = lgb_tra)'''

            #gbm = lgb.train(param, lgb_tra)
            #preds = gbm.predict(val_x)
            #d = mean_absolute_error(val_y, preds)
            regr.fit(tra_x, tra_y) #学習用データを渡してモデルの学習を行う
            y_reg = regr.predict(val_x) #テストデータの全てで予測する'''
            d = mean_absolute_error(val_y, y_reg)
            ab.append(d)

        #score = cross_val_score(regr, x_train, y_train, groups = group, scoring = "neg_mean_absolute_error", cv = logo, n_jobs = -1)
        #ab_mean = score.mean()
        ab_mean = np.mean(ab)
        print(ab_mean)

        return ab_mean

    #optuna
    study = optuna.create_study()
    study.optimize(objective, n_trials = 100)
    best = [study.best_params['gamma'], study.best_params['C'], study.best_params['epsilon']]
    #best = [study.best_params['lambda_l1'], study.best_params['lambda_l2'], study.best_params['num_leaves'], study.best_params['feature_fraction'], study.best_params['bagging_fraction'], study.best_params['bagging_freq'], study.best_params['min_child_samples']]

    path = os.path.expanduser("W:\\Crowdsourcing")
    path += "\\loggerstation_iwatsuru"
    path += "\\result"
    path += "\\MAE"
    out_path = os.path.join(path, "SVR")

    if not os.path.exists(os.path.join(out_path, "parameter_svr_sc_10_model.csv")):
        with open(os.path.join(out_path, 'parameter_svr_sc_10_model.csv'), 'a') as f:
            writer = csv.writer(f, lineterminator = '\n')
            header = ["gamma", "C", "epsilon"]
            #header = ["lambda_l1", "lambda_l2", "num_leaves", "feature_fraction","bagging_fraction", "bagging_freq", "min_child_samples"]
            writer.writerow(header)

    with open(os.path.join(out_path, 'parameter_svr_sc_10_model.csv'), 'a') as f:
        writer = csv.writer(f, lineterminator = '\n')
        writer.writerow(best)

    #チューニングしたハイパーパラメーターをフィット
    optimized_regr = SVR(kernel = 'rbf', gamma = study.best_params['gamma'], C = study.best_params['C'], epsilon = study.best_params['epsilon'])
    #optimized_regr = SVR(kernel = study.best_params['kernel'] , gamma = study.best_params['gamma'], C = study.best_params['C'], epsilon = study.best_params['epsilon'])
    #optimized_regr.fit(x_train, y_train)

    '''lgb_train = lgb.Dataset(x_train, y_train)
    lgb_test = lgb.Dataset(x_test, y_test, reference = lgb_train)

    gbm = lgb.train({'objective' : 'regression',
            'metric' : 'l1',
            'lambda_l1' : study.best_params['lambda_l1'],
            'lambda_l2' : study.best_params['lambda_l2'],
            'num_leaves' : study.best_params['num_leaves'],
            'feature_fraction' : study.best_params['feature_fraction'],
            'bagging_fraction' : study.best_params['bagging_fraction'],
            'bagging_freq' : study.best_params['bagging_freq'],
            'min_child_samples' : study.best_params['min_child_samples']
            }, lgb_train)
    preds = gbm.predict(x_test)
    d = mean_absolute_error(y_test, preds)'''

    #最適パラメータの表示と保持
    best_params = study.best_trial.params
    best_score = study.best_trial.value
    print(f'最適パラメータ {best_params}\nスコア {best_score}')

    #y_reg = optimized_regr.predict(x_test) #テストデータの全てで予測する

    #d = mean_absolute_error(y_test, y_reg)
    #d = np.sqrt(mean_squared_error(y_test, y_reg))
    #ans.append(d)

    #return ans


#関数select
def select(x, y, count_x, num_examinee, num_feature, questnum, k_feature): #引数はリストx, y, count_xの他に、ユーザの数を表す変数num_examinee，特徴の数を表す(xの列の数)変数num_feature，一度に分析する問題数を表すquestnumである．

    ans = [] #平均絶対誤差の値を格納するためのリストans
    for i in range(num_examinee): #0～9の値でループ
        #classifierを用いて学習モデルを作る
        clf = classifier.set_classifier("svr", i + 1)

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

        group = []
        count = 0
        for j in range(num_examinee):
            if (i != j):
                for k in range(count_x[j], count_x[j + 1]):
                    group.append(count)
            count += 1
        #print(group)
        logo = LeaveOneGroupOut()
        cv_gen = logo.split(x_train, y_train, groups = group)
        cv = list(cv_gen)

        fs = sfs(clf, k_features = k_feature, forward = True, floating = True, verbose = 2, scoring = 'neg_mean_absolute_error', cv = cv, n_jobs = -1)
        fs = fs.fit(x_train, y_train)
        print(fs.k_feature_idx_, fs.k_score_)
        selection = [0] * 29
        for k in fs.k_feature_idx_:
            selection[k] = 1
        selection.append(fs.k_score_)

        path = os.path.expanduser("W:\\Crowdsourcing")
        path += "\\loggerstation_iwatsuru"
        path += "\\result"
        path += "\\MAE"
        path += "\\SVR"
        out_path = os.path.join(path, "selection")

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        if not os.path.exists(os.path.join(out_path, "feature_select_" + str(k_feature) + ".csv")):
            with open(os.path.join(out_path, "feature_select_" + str(k_feature) + ".csv"), 'a') as f:
                writer = csv.writer(f, lineterminator = '\n')
                header = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "score"]
                writer.writerow(header)

        with open(os.path.join(out_path, "feature_select_" + str(k_feature) + ".csv"), 'a') as f:
            writer = csv.writer(f, lineterminator = '\n')
            writer.writerow(selection)

        x_train_fs = fs.transform(x_train)
        x_test_fs = fs.transform(x_test)

        clf.fit(x_train_fs, y_train) #学習用データを渡してモデルの学習を行う
        y_reg = clf.predict(x_test_fs) #テストデータの全てで予測する

        '''d = 0
        for j in range(len(y_reg)): #len(y_reg)は各人が回答した問題数
            d += abs(y_reg[j] - y_test[j]) #推定したものと正解とで誤差の絶対値の合計を求めている.y_reg:推定結果, y_test:正解．
        d /= len(y_reg) #求めた誤差の合計を問題数で除算することで，平均絶対誤差を求めている'''

        d = mean_absolute_error(y_test, y_reg)
        #d = np.sqrt(mean_squared_error(y_test, y_reg))
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

    for p in range(1): #問題数
        result_temp = []
        questnum = p + 10
        print(questnum)

        #関数analyzeの処理
        x, y, count_x = analyze(questnum, use_file)

        y = np.array(y) #yをnp.array型に変換
        num_feature = x.shape[1] #リストxの列数を変数num_featureに格納

        tuning(x, y, count_x, len(use_file), num_feature, questnum) #ansは平均絶対誤差の値が格納されたリストが返ってくる
        '''result_temp.append(questnum) #一度に分析する数をリストresult_tempに格納

        sum = 0
        for i in range(len(ans)):
            result_temp.append(ans[i]) #リストresult_tempにリストansの各値を追加する
            sum += ans[i]
        result_temp.append(sum / len(ans)) #リストresult_tempの末尾に全員の平均絶対誤差の平均値を格納
        print(result_temp)'''

        '''out_path = os.path.join(out_path, "selection")
        for k_feature in range(28):
            result_temp = []
            ans = select(x, y, count_x, len(use_file), num_feature, questnum, k_feature + 1)
            result_temp.append(k_feature + 1)
            sum = 0
            for i in range(len(ans)):
                result_temp.append(ans[i]) #リストresult_tempにリストansの各値を追加する
                sum += ans[i]
            result_temp.append(sum / len(ans)) #リストresult_tempの末尾に全員の平均絶対誤差の平均値を格納
            print(result_temp)

            if not os.path.exists(os.path.join(out_path, "selection_result.csv")):
                with open(os.path.join(out_path, 'selection_result.csv'), 'a') as f:
                    writer = csv.writer(f, lineterminator = '\n')
                    #header = ["k_feature", "guest", "pre01", "pre02", "p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09", "p10", "Ave"]
                    header = ["k_feature", "p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09", "p10", "Ave"]
                    writer.writerow(header)

            with open(os.path.join(out_path, 'selection_result.csv'), 'a') as f:
                writer = csv.writer(f, lineterminator = '\n')
                writer.writerow(result_temp)'''


if __name__ == "__main__":
	main()