# -*- coding : utf-8 -*-

#説明###
#
#user, path, tobii_pro_pathを変更する
#user, pathは自分のパソコンでのファイル名などに変更
#
#"001_back.png"を同一フォルダに入れておく必要がある
# 
#38, 39行目の加算する値を変更していってズレの値を求める
#
##########


import sys
import warnings
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


user = "\\p05\\"
if not sys.warnoptions:
    warnings.simplefilter("ignore")

path = os.path.expanduser("W:\\Crowdsourcing")
path += "\\loggerstation_iwatsuru"

#要変更
path += user
input_dirs = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
input_dirs.sort()
#print(os.listdir(path))

#座標のズレを調節する値
X_sa = 0 #右に動かす場合Xはプラス
Y_sa = 0 #下に動かす場合Yはプラス

mag = 1.0 #倍率

print(user)
for input_dir in input_dirs:
    fig = plt.figure(figsize = (19.20, 10.80))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(1, 1, 1)
    print(input_dir)
    #plt.figure()
    section_path = os.path.join(path, input_dir)
    if os.path.exists(os.path.join(section_path, "output")):
        fixation_files = glob.glob(os.path.join(section_path, "output\\*_fixations.csv"))
        fixation_files.sort()

        count = 0
        for fixation_path in fixation_files:
            tobii_df = pd.read_csv(fixation_path)
            id_fixation = tobii_df["#timestamp"].values.tolist()
            x_fixation = tobii_df["fixation_x"].values.tolist()
            y_fixation = tobii_df["fixation_y"].values.tolist()

            x = []
            y = []
            for i in range(len(x_fixation)):
                x.append(x_fixation[i] * mag)
                y.append(y_fixation[i] * mag)

            ax.scatter(x, y, label = "section(" + str(count + 1) + ")")
            plt.title("Gaze behavior")

            ax.set_xlim([0, 1920])
            ax.set_ylim([1080, 0])
            #グリッド線を表示するならTrue
            ax.grid(True)
            ax.legend(bbox_to_anchor = (1, 1.05), loc = 'upper right', borderaxespad = 0, fontsize = 8)

            image_path = os.path.join(section_path, "001_back.png")
            im = Image.open(image_path)

            im = im.resize((im.width, im.height), Image.LANCZOS)
            ax.imshow(im)
            #plt.savefig(user + "section(" + str(count + 1) + ")_graph.png")

            count += 1

plt.show()