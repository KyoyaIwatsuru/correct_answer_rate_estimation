# -*- coding: utf-8 -*-

###
#
#featureファイルの編集を行う
#
###


import sys
import warnings
import os
import glob
import pandas as pd
import numpy as np


def insert_df_middle_row(idx, df_original, df_insert):
    return pd.concat([df_original[:idx], df_insert, df_original[idx:]]).reset_index(drop = True)


def fetch(section_path, i, j):
    fixation_files = glob.glob(os.path.join(section_path, "output\\*_fixations.csv"))
    fixation_files.sort()
    confidence_answer_path = os.path.join(section_path, "confidence_answer.csv")

    conf_df = pd.read_csv(confidence_answer_path)
    begin_time = conf_df["begin_timestamp"].values.tolist()
    end_time = conf_df["end_timestamp"].values.tolist()
    fixation_dfi = pd.read_csv(fixation_files[i]).dropna()
    fixation_dfj = pd.read_csv(fixation_files[j]).dropna()

    dfi = fixation_dfi[(begin_time[i] < fixation_dfi["#timestamp"]) & (fixation_dfi["#timestamp"] < end_time[i])]
    fixations_i = dfi.values.tolist()
    dfj = fixation_dfj[(begin_time[i] < fixation_dfj["#timestamp"]) & (fixation_dfj["#timestamp"] < end_time[i])]
    fixations_j = dfj.values.tolist()

    x = []
    y = []
    for k in range(len(fixations_i)):
        x.append(float(fixations_i[k][1])) #fixation_x
        y.append(float(fixations_i[k][2])) #fixation_y
    for k in range(len(fixations_j)):
        x.append(float(fixations_j[k][1])) #fixation_x
        y.append(float(fixations_j[k][2])) #fixation_y
    std_x = np.std(x) #13:x座標の分散値
    std_y = np.std(y) #14:y座標の分散値

    return std_x, std_y


def main():
    use_file = ["\\guest\\", "\\pre01\\", "\\pre02\\", "\\p01\\", "\\p02\\", "\\p03\\", "\\p04\\", "\\p05\\", "\\p06\\", "\\p07\\", "\\p08\\", "\\p09\\", "\\p10\\"]

    for user in use_file:
        print(user)
        if not sys.warnoptions:
            warnings.simplefilter("ignore")

        path = os.path.expanduser("W:\\Crowdsourcing")
        path += "\\loggerstation_iwatsuru"

        #要変更
        path += user
        input_dirs = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
        input_dirs.sort()

        count = 0

        for input_dir in input_dirs:
            #print(input_dir)
            section_path = os.path.join(path, input_dir)
            feature_path = os.path.join(section_path, "output\\feature.csv")

            if not os.path.exists(feature_path):
                continue
            feature_df = pd.read_csv(feature_path)
            #print(feature_df)
            id = feature_df["#id"].values.tolist()
            user_id = feature_df["user_id"].values.tolist()
            document_id = feature_df["document_id"].values.tolist()
            question_id = feature_df["question_id"].values.tolist()
            choice = feature_df["choice"].values.tolist()
            label = feature_df["label"].values.tolist()
            #print(question_id)
            F = []
            for i in range(30):
                F.append(feature_df["f" + str(i + 1)].values)
            #print(F)

            for i in range(len(question_id)):
                for j in range(i + 1, len(question_id)):
                    if question_id[i] == question_id[j]:
                        print(input_dir, i, j) #被り調査
                        std_x, std_y = fetch(section_path, i, j)

                        f1 = F[0][i] + F[0][j] #1:選択肢を見たfixationの総数
                        f2 = F[1][i] + F[1][j] #2:問題文を見たfixationの総数
                        if f1 == 0 and f2 == 0:
                            f3 = 0
                            f4 = 0
                        else:
                            f3 = 1. * f1 / (F[16][i] + F[16][j] + 2) #3:全体のうち選択肢のfixationの割合
                            f4 = 1. * f2 / (F[16][i] + F[16][j] + 2) #4:全体のうち問題文のfixationの割合
                        f5 = F[4][i] + F[4][j] #5:選択肢のfixation持続時間の合計
                        f6 = f5 / f1 #6:選択肢のfixation持続時間の平均
                        f7 = max(F[6][i], F[6][j]) #7:選択肢のfixation持続時間の最大値
                        f8 = min(F[7][i], F[7][j]) #8:選択肢のfixation持続時間の最小値
                        f9 = F[8][i] + F[8][j] #9:問題文のfixation持続時間の合計
                        f10 = f9 / f2 #10:問題文のfixation持続時間の平均
                        f11 = max(F[10][i], F[10][j]) #11:問題文のfixation持続時間の最大値
                        f12 = min(F[11][i], F[11][j]) #12:問題文のfixation持続時間の最小値
                        f17 = F[16][i] + F[16][j] #17:saccadeの回数
                        f13 = std_x #13:x座標の分散値(数字はxの平均値)
                        f14 = std_y #14:y座標の分散値(数字はyの平均値)
                        f15 = F[14][i] + F[14][j] #15:saccadeの距離の合計
                        f16 = f15 / f17 #16:saccadeの距離の平均
                        f18 = F[17][i] + F[17][j] #18:選択肢間のsaccadeの回数
                        f19 = F[18][i] + F[18][j] #19:問題文内のsaccadeの回数
                        f20 = F[19][i] + F[19][j] #20:選択肢-問題間のsaccadeの回数
                        f21 = F[20][i] + F[20][j] #21:saccade持続時間の合計
                        f22 = f21 / f17 #22:saccade持続時間の平均
                        f23 = max(F[22][i], F[22][j]) #23:saccade持続時間の最大値
                        f24 = min(F[23][i], F[23][j]) #24:saccade持続時間の最小値
                        f25 = F[24][i] + F[24][j] #25:saccade時の視点の速度の合計
                        f26 = f25 / f17 #26:saccade時の視点の速度の平均
                        f27 = max(F[26][i], F[26][j]) #27:saccade時の視点の速度の最大値
                        f28 = min(F[27][i], F[27][j]) #28:saccade時の視点の速度の最小値
                        f29 = F[28][i] #29:解答時間
                        f30 = F[29][i] #30:解答の正誤
                        feature_df = feature_df.drop(feature_df.index[[i, j]])
                        #print(feature_df)
                        f_df = pd.DataFrame({'#id' : [id[i]], 'user_id' : [user_id[i]], 'document_id' : [document_id[i]], 'question_id' : [question_id[i]], 'choice' : [choice[i]], 'label' : [label[i]]})
                        for k in range(30):
                            f_df["f" + str(k + 1)] = eval("f" + str(k + 1))
                        #print(f_df)
                        feature_df = insert_df_middle_row(i, feature_df, f_df)
                        #print(feature_df)
                        feature_df.to_csv(feature_path, index = False)


if __name__ == "__main__":
	main()