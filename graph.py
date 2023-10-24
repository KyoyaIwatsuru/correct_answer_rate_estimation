import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = os.path.expanduser("W:\\Crowdsourcing\\loggerstation_iwatsuru\\result\\MAE\\SVR\\selection")
feature_files = glob.glob(os.path.join(path, "feature_select_*.csv"))
feature_files.sort()
j = 0
for feature_path in feature_files:
    feature_df = pd.read_csv(feature_path)
    #print(np.sum(feature_df.iloc[:, 0]))

    temp_x = []
    temp_y = []
    for i in range(29):
        temp_x.append(i + 1)
        temp_y.append(np.sum(feature_df.iloc[:, i]))
    #print(type(temp_x), type(temp_y))

    left = np.array(temp_x)
    height = np.array(temp_y)
    #print(type(left), type(height))
    plt.xticks(temp_x, rotation = 90)

    plt.bar(left, height)
    #plt.show()

    plt.savefig(os.path.join(path, str(j + 1) + ".png"))
    j += 1