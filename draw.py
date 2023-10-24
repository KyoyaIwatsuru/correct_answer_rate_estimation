# -*- coding: UTF-8 -*-


import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


#関数draw_results
def draw_results(window_size, error, time, answer, out_path):
    #plot main result
    '''error = np.sum(error, axis = 1) / 9
    time = np.sum(time, axis = 1) / 9
    answer = np.sum(answer, axis = 1) / 9'''
    fig, ax = plt.subplots()
    #ax.set_title('平均絶対誤差の問題数による変化', fontname = 'MS Gothic')
    ax.set_ylabel('平均絶対誤差', fontname = 'MS Gothic')
    ax.set_xlabel('問題数', fontname = 'MS Gothic')
    ax.grid()
    ax.plot(window_size, error, color = 'red', label = '統計的特徴量')
    #ax.plot(window_size, time, color = 'blue', label = '解答時間')
    ax.plot(window_size, answer, color = 'green', label = '平均正答率')
    #ax.plot(window_size, nawa, color = 'black', label = 'conventional method')
    #ax.plot(window_size, error, label = ["p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09", "p10"])
    ax.legend(loc = 'upper right', prop = {'family' : 'MS Gothic'})
    plt.savefig(os.path.join(out_path, "final_result_sc_10_model_jp_pptx.png"), dpi = 300)

def main():
    path = os.path.expanduser("W:\\Crowdsourcing")
    path += "\\loggerstation_iwatsuru"
    path += "\\result"
    path += "\\MAE"
    out_path = os.path.join(path, "SVR")
    lgb_path = os.path.join(path, "LGB")

    result_path = os.path.join(out_path, "final_result_sc_10_model.csv")
    time_path = os.path.join(out_path, "final_result_time.csv")
    answer_path = os.path.join(out_path, "final_result_answer.csv")
    nawa_path = os.path.join(out_path, "shozemi_result_SVM_not_normalized_4_features.csv")
    feature_df = pd.read_csv(result_path)
    time_df = pd.read_csv(time_path)
    answer_df = pd.read_csv(answer_path)
    nawa_df = pd.read_csv(nawa_path)
    result_final = feature_df.values
    time_final = time_df.values
    answer_final = answer_df.values
    nawa_final = nawa_df.values
    window_size = result_final[:, 0] #window_size：ウィンドウサイズ
    error = result_final[:, 11] #error：平均絶対誤差(10人の平均)
    time = time_final[:, 11]
    answer = answer_final[:, 11]
    nawa = nawa_final[:, 11]
    #print(window_size)
    #print(error)

    draw_results(window_size, error, time, answer, out_path) #draw_resultsの処理で結果を描画する

if __name__ == "__main__":
	main()