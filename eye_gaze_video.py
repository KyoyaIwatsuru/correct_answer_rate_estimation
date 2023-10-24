#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import animatplot as amp
from PIL import Image


#グラフの描画
def plot_animation(section_path, ref_df, begin_time, end_time):
    #refの経路描画
    ref_df = ref_df[(begin_time < ref_df["#timestamp"]) & (ref_df["#timestamp"] < end_time)]
    initial = np.array(ref_df["#timestamp"])[0]
    time_data_np = (np.array(ref_df["#timestamp"]) - initial) / 1000
    x_np = np.array(ref_df["gaze_x"])
    y_np = np.array(ref_df["gaze_y"])

    Xs_log = np.array([x_np[t : t + 100] for t in range(len(time_data_np) - 100)]) #X軸データ × 時間軸 分の配列
    Ys_log = np.array([y_np[t : t + 100] for t in range(len(time_data_np) - 100)]) #Y軸データ × 時間軸 分の配列
    Time_log = np.array([time_data_np[t : t + 100] for t in range(len(time_data_np) - 100)])

    #subplotの描画 (X-Yの情報を3行分の画面で表示)
    fig, ax1 = plt.subplots()
    #ax2 = plt.subplot2grid((3,2), (0,1))
    #ax3 = plt.subplot2grid((3,2), (1,1))
    #ax4 = plt.subplot2grid((3,2), (2,1))
    fig.patch.set_facecolor('white')

    ax1.set_xlim(0, 1920) #描画範囲の設定
    ax1.set_ylim(0, 1080) #描画範囲の設定
    ax1.invert_yaxis()

    image_path = os.path.join(section_path, "001_back.png")
    im = Image.open(image_path)
    ax1.imshow(im, aspect = 'auto')

    block1 = amp.blocks.Scatter(Xs_log, Ys_log, label = "eye_gaze", ax = ax1)
    fig.legend()

    Time = amp.Timeline(Time_log[:, 0], units = 's', fps = 100)
    anim = amp.Animation([block1], Time)
    anim.controls()
    video_path = os.path.join(section_path, "eye_gaze_video.mp4")
    #anim.save_gif(video_path) #gifならeye_gaze_videoとする
    anim.save(video_path)

    #plt.show()



if __name__ == '__main__':

    user = "\\p06\\"
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    path = os.path.expanduser("W:\\Crowdsourcing")
    path += "\\loggerstation_iwatsuru"

    path += user
    #input_dirs = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
    #input_dirs.sort()
    input_dirs = ['01']

    count = 0

    for input_dir in input_dirs:
        section_path = os.path.join(path, input_dir)
        csv_file_path = os.path.join(section_path, "tobii_pro_gaze_edit.csv")
        answer_path = os.path.join(section_path, "confidence_answer.csv")

        #CSVの読み込み
        ref_df = pd.read_csv(csv_file_path, encoding = "utf-8-sig") #日本語データ(Shift-Jis)を含む場合を想定
        answer_df = pd.read_csv(answer_path)
        begin_time = answer_df["begin_timestamp"].values[0]
        end_time = answer_df["end_timestamp"].values[0]
        plot_animation(section_path, ref_df, begin_time, end_time)