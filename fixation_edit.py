# -*- coding: utf-8 -*-

###
#
#視線情報を平行移動させるファイル
#基本的に，mapping.pyの後に使用する
#
###


import sys
import warnings
import os
import pandas as pd


use_file = ["\\guest\\", "\\pre01\\", "\\pre02\\", "\\p01\\", "\\p02\\", "\\p03\\", "\\p04\\", "\\p05\\", "\\p06\\", "\\p07\\", "\\p08\\", "\\p09\\", "\\p10\\"]

count = 0

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

    #座標のズレを調節する値, mapping.pyで求めた値
    #X_sa = [0] #右に動かす場合Xはプラス
    #Y_sa = [0] #上に動かす場合Yはプラス

    for input_dir in input_dirs:
        mag = [1.16, 1.25, 1.18, 1.32, 1.15, 1.40, 1.35, 1.27, 1.32, 1.20, 1.32, 1.35, 1.18]

        if (user == "\\pre01\\" and (int(input_dir) == 25 or int(input_dir) == 26)):
            mag[count] = 1.35
        if (user == "\\p02\\" and (int(input_dir) >= 13 and int(input_dir) <= 16)):
            mag[count] = 1.10
        if (user == "\\p02\\" and (int(input_dir) == 1 or int(input_dir) >= 17)):
            mag[count] = 1.24
        if (user == "\\p03\\" and int(input_dir) >= 12):
            mag[count] = 1.35
        if (user == "\\p04\\" and int(input_dir) >= 17):
            mag[count] = 1.30
        if (user == "\\p05\\" and (int(input_dir) >= 28 and int(input_dir) <= 35)):
            mag[count] = 1.45
        if (user == "\\p05\\" and int(input_dir) >= 36):
            mag[count] = 1.20
        if (user == "\\p08\\" and int(input_dir) >= 16):
            mag[count] = 1.40

        #print(input_dir)
        #plt.figure()
        section_path = os.path.join(path, input_dir)
        tobii_pro_path = os.path.join(section_path, "tobii_pro_gaze.csv")

        tobii_df = pd.read_csv(tobii_pro_path)
        tobii_df.fillna('NaN')

        x = tobii_df["gaze_x"].values.tolist()
        y = tobii_df["gaze_y"].values.tolist()
        X = []
        Y = []
        for i in range(len(x)):
            X.append(x[i] * mag[count])
            Y.append(y[i] * mag[count])

        tobii_df["gaze_x"] = X
        tobii_df["gaze_y"] = Y
        #print(list(tobii_df.columns.values))

        edit_path = os.path.join(section_path, "tobii_pro_gaze_edit.csv")
        tobii_df.to_csv(edit_path, index = False, na_rep = 'nan')

        #head = ["#id", "begin_datetime", "begin_timestamp", "end_timestamp", "user_id", "document_id", "question_id", "choice", "confidence_predicted", "correctness", "class_label"]

        #if not os.path.exists(os.path.join(output_dir, "confidence_answer_label.csv")):
        #   with open(os.path.join(output_dir, "confidence_answer_label.csv"), "a") as f:
        #       writer = csv.writer(f, lineterminator = "\n")
        #       writer.writerow(head)

        #with open(os.path.join(output_dir, "confidence_answer_label.csv"), "a") as f:
        #   writer = csv.writer(f, lineterminator = "\n")
        #   writer.writerows(answer_df.values)

    count += 1