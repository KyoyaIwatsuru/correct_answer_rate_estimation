#説明###
#
#user, pathを変更する
#user, pathは自分のパソコンでのファイル名などに変更
#
##########


import sys
import warnings
import os
import shutil
from progressbar import ProgressBar #(pip install progressbar2)
import pandas as pd
import numpy as np
import math
#from scipy.ndimage import filters
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import matplotlib.cm as cm


#os.chdir(os.path.dirname(__file__))


#関数isMinimumFixation（引数はリストX = X_, Y = Y_, 変数mfx = min_fixation_size( = 50))
#minimum fixationに含まれるかの判定（含まれるならTrue, 含まれないならFalseを返す）
def isMinimumFixation(X, Y, mfx):
    if max([max(X) - min(X), max(Y) - min(Y)]) < mfx:
        return True
    return False


#関数detectFixations（引数は、リストtime = timestamps, X = gaze_x, Y = gaze_y 及び、save_path(保存場所のパス))
def detectFixations(times, X, Y, min_concat_gaze_count = 9, min_fixation_size = 50, max_fixation_size = 80, save_path = ""): #min_fixation_size : minimum fixationとみなせる範囲，max_fixation_size : large fixationとみなせる範囲
    fixations = [] #fixationに関するデータを格納するためのリスト
    i = 0
    j = 0

    while max([i, j]) < len(times) - min_concat_gaze_count: #視線座標のデータが9(min_concat_gaze_count)個分とれなくなったらこのループは終了
        #視線の座標データを9個ずつ精査している(0.01sごとに区切るため)
        X_ = list(X[i : i + min_concat_gaze_count])
        Y_ = list(Y[i : i + min_concat_gaze_count])

        #関数isMinimumFixationの処理（引数はリストX_, Y_, 変数min_fixation_size)
        #isMinimumFixationの返り値がTrueの場合は以下のif分の処理．Falseならばiを+1して視線の座標データを9個とってminimum fixationに含まれるか精査するところからやり直し．
        if isMinimumFixation(X_, Y_, min_fixation_size):
            j = i + min_concat_gaze_count
            c = 0
            begin_time = times[i]
            end_time = times[j - 1]

            while(c < min_concat_gaze_count and j < len(times)): #large fixation(max_fixation_size)に含まれない点の個数(c)が、連続して9(min_concat_gaze_count)個現れると終了
                X_.append(X[j])
                Y_.append(Y[j])
                #print(X_)
                if max([max(X_) - min(X_), max(Y_) - min(Y_)]) > max_fixation_size: #large fixationに含まれないとき
                    if c == 0:
                        i = j
                    X_.pop()
                    Y_.pop()
                    c += 1
                else:
                    c = 0
                    end_time = times[j]
                j += 1
            i = i - 1
            j = j - 1
            fixations.append([begin_time, np.mean(X_), np.mean(Y_), end_time - begin_time]) #リストfixationsに一番最初のデータの時刻、large fixationの範囲内に含まれる視線情報のx座標, y座標の平均、large fixationの範囲内に含まれるデータの所要時間を追加
            #if((end_time - begin_time) == 0):
                #print("end_time - begin_time = 0")
        i += 1

    if len(fixations) < 2:
        #print("len(fixations) < 2")
        return np.array([])

    lengths = [0] #仮入れ
    angles = [0] #仮入れ
    durations = [1] #仮入れ
    for i in range(1, len(fixations)):
        #fixationのx座標の変位，y座標の変位，時間の変位を求める
        delta_x = fixations[i][1] - fixations[i-1][1]
        delta_y = fixations[i][2] - fixations[i-1][2]
        delta_t = fixations[i][0] - (fixations[i-1][0] + fixations[i-1][3])
        #if (delta_t == 0):
        #print("delta_t = 0")

        #2つのfixation間の距離, 角度の変化, 時間の変位をそれぞれリストlengths, angles, durationsに値を追加
        lengths.append(math.sqrt(delta_x * delta_x + delta_y * delta_y))
        angles.append(math.atan2(delta_y, delta_x))
        durations.append(delta_t)

    #リストlengths, angles, 速度(lengths/durations)の値が縦に連結され最後に転置する．それをsaccadesとする．
    saccades = np.vstack((lengths, angles, np.array(lengths) / np.array(durations))).T
    #fixationsの値(4要素)とsaccadesの値(3要素)を横方向に結合
    results = np.hstack((fixations, saccades))
    #print(len(fixations))
    #print(len(saccades))
    #print(len(results))
    #print("")

    #csvファイルとして保存
    if save_path != "":
        df = pd.DataFrame(results, columns = ["#timestamp", "fixation_x", "fixation_y", "fixation_duration", "saccade_length", "saccade_angle", "saccade_velocity"])
        df = df.replace([np.inf, -np.inf], np.nan)
        #print(df.isnull().sum())
        df = df.dropna()
        df.to_csv(save_path, index = False)
    return results


#def plotScanPath(
#       X, Y, durations, figsize = (30, 15),
#       bg_image = "", save_path = "", halfPage = False):
#   plt.figure(figsize = figsize)
#   if bg_image != "":
#       img = mpimg.imread(bg_image)
#       plt.imshow(img)
#       if halfPage:
#           plt.xlim(150, 1000)
#       else:
#           plt.xlim(0, len(img[0]))
#       plt.ylim(len(img), 0)
#   scale = float(figsize[0]) / 40.0

#   plt.plot(X, Y, "-", c = "blue", linewidth = scale, zorder = 1, alpha = 0.8)
#   plt.scatter(X, Y, durations * scale, c = "b", zorder = 2, alpha = 0.3)
#   plt.scatter(X[0], Y[0], durations[0] * scale, c = "g", zorder = 2, alpha = 0.6)
#   plt.scatter(X[-1], Y[-1], durations[-1] * scale, c = "r", zorder = 2, alpha = 0.6)

#   if save_path != "":
#       plt.tick_params(labelbottom = "off")
#       plt.tick_params(labelleft = "off")
#       plt.savefig(save_path, bbox_inches = "tight", pad_inches = 0.0)
#       plt.close()


#def plotHeatmap(
#       X, Y, durations, figsize = (30, 15),
#       bg_image = "", save_path = "", data_save_path = ""):
#   values = calcHeatmap(
#           X, Y, durations, figsize, bg_image, save_path, data_save_path)
#   plotHeatmapFromExported(
#           values, figsize, bg_image, save_path, data_save_path)


#def calcHeatmap(
#       X, Y, durations, figsize = (30, 15),
#       bg_image = "", save_path = "", data_save_path = ""):
#   if bg_image != "":
#       img = mpimg.imread(bg_image)

#   gx, gy = np.meshgrid(np.arange(0, len(img[0])), np.arange(0, len(img)))
#   values = np.zeros((len(img), len(img[0])))

#   for i in range(len(X)):
#       if X[i] < len(img[0]) and Y[i] < len(img):
#           values[int(Y[i]), int(X[i])] += durations[i]
#   values = filters.gaussian_filter(values, sigma = 50)

#   return values / np.max(values)


#def plotHeatmapFromExported(
#       values, figsize = (30, 15), bg_image = "",
#       save_path = "", data_save_path = ""):
#   plt.figure(figsize = figsize)
#   if bg_image != "":
#       img = mpimg.imread(bg_image)
#       plt.imshow(img)
#       plt.xlim(0, len(img[0]))
#       plt.ylim(len(img), 0)

#   masked = np.ma.masked_where(values < 0.05, values)
#   cmap = cm.jet
#   cmap.set_bad('white', 1.)
#   plt.imshow(masked, alpha = 0.4, cmap = cmap)

#   if save_path != "":
#       plt.tick_params(labelbottom = "off")
#       plt.tick_params(labelleft = "off")
#       plt.savefig(save_path, bbox_inches = "tight", pad_inches = 0.0)
#       plt.close()

#   if data_save_path != "":
#       np.savetxt(data_save_path, values, delimiter = ",", fmt = "%f")


#def p(dir_name):
#   return os.path.abspath(os.path.expanduser(dir_name))


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

        for input_dir in input_dirs:
            section_path = os.path.join(path, input_dir)
            #tobii_pro_path = os.path.join(section_path, "tobii_pro_gaze.csv") #fixation_edit前
            tobii_pro_path = os.path.join(section_path, "tobii_pro_gaze_edit.csv") #fixation_edit後
            tobii_4c_path = os.path.join(section_path, "tobii_gaze.csv")

            if os.path.exists(tobii_pro_path): #該当フォルダ内にtobii_pro_gaze_cut.csvがあるとき
                gaze_df = pd.read_csv(tobii_pro_path).dropna() #gaze_dfは欠損値を除去した場合
                gaze_df_ = pd.read_csv(tobii_pro_path) #gaze_df_は欠損値を除去せず，欠損値ごと含めた場合
            elif os.path.exists(tobii_4c_path): #該当フォルダ内にtobii_pro_gaze_cut.csvがなく，tobii_gaze.csvがあるとき
                gaze_df = pd.read_csv(tobii_4c_path).dropna()
                gaze_df_ = pd.read_csv(tobii_4c_path)
            else:
                print("\n[\033[31mERROR\033[0m] Invalid directory name.\n")
                main()

            """on_surface = input("Is this data recorded on Surface Studio? (y/N): ")
            if on_surface in ["y", "Y", "yes", "Yes", "YES"]:
                scale = 2.0
            else:
                scale = 1.0"""
            scale = 1.0

            output_dir = os.path.join(section_path, 'output')
            if os.path.isdir(output_dir):
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
            else:
                os.makedirs(output_dir)

            #capture.csvファイルの読み込み
            captures_df = pd.read_csv(os.path.join(section_path, "capture.csv")) #edit
            captures = captures_df["#timestamp"].values.tolist() #tolistで#timestampの値をリストcapturesへ変換
            gaze = gaze_df["#timestamp"].values #tobii_pro_gaze_cut.csvファイルの#timestamp列の値をリストgazeに格納
            #print(gaze.size)
            if gaze.size != 0:
                captures.append(gaze_df["#timestamp"].values[-1]) #gaze_dfの#timestamp列の最後尾の値をリストcapturesに格納
                #print(len(captures))

            size = len(captures) - 1
            #print(input_dir, size - 1)
            pbar = ProgressBar(max_value = size)
            for i in range(1, size): #ループは問題1問ごとに回っている
                pbar.update(i + 1)
                #gaze_dfあるいはgaze_df_のデータに対し，capture.csvの時間(#timestamp)を基にどの問題を解いている際のデータか判別しdfあるいはdf_に代入
                df = gaze_df[(captures[i - 1] < gaze_df["#timestamp"]) & (gaze_df["#timestamp"] < captures[i])]
                df_ = gaze_df_[(captures[i - 1] < gaze_df_["#timestamp"]) & (gaze_df_["#timestamp"] < captures[i])]
                #print(df)

                timestamps = df["#timestamp"].values
                #欠損値を除去した場合の視線のx,y座標の情報
                gaze_x = df["gaze_x"].values * scale
                gaze_y = df["gaze_y"].values * scale
                #欠損値を除去しておらず，欠損値を含めた場合の視線のx,y座標の情報
                gaze_x_ = df_["gaze_x"].values * scale
                gaze_y_ = df_["gaze_y"].values * scale

                if len(timestamps) == 0:
                    #print("len(timestamps) = 0")
                    continue

                #if len(gaze_x) < len(gaze_x_) * 0.8:
                    #print(i, "len(gaze_x) < len(gaze_x_) * 0.8")
                    #continue

                #関数detectFixationsの処理（引数は、リストtimestamps, リストgaze_x, リストgaze_y及び、保存場所のパス）
                fx = detectFixations(timestamps, gaze_x, gaze_y, save_path = os.path.join(output_dir, str(i).zfill(3) + "_fixations.csv")) #edit

                if len(fx) == 0:
                    #print("len(fx) = 0")
                    continue

                """plotScanPath(fx[:, 1], fx[:, 2], fx[:, 3],
                            bg_image = os.path.join(input_dir, str(i + 1).zfill(3) + "_back.png"),
                            save_path = os.path.join(output_dir, "new_" + str(i).zfill(3) + "_scanpath.png")) #edit

                plotHeatmap(fx[:, 1], fx[:, 2], fx[:, 3],
                            bg_image = os.path.join(input_dir, str(i + 1).zfill(3) + "_back.png"),
                            save_path = os.path.join(output_dir, "new_" + str(i).zfill(3) + "_heatmap.png")) #edit"""

            print("\n[\033[32mDONE\033[0m] Exported to " + output_dir + "\n")


if __name__ == '__main__':
    main()