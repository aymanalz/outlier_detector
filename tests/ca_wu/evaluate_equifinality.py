import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

figures_folder = r"results\figures"


files = ['ca_wu_od_1253.dat','ca_wu_od_2130.dat','ca_wu_od_5523.dat', 'ca_wu_od_8891.dat', 'ca_wu_od_52347.dat']

width_in = 7
height_in = 6
fig = plt.figure(figsize=(width_in, height_in))


for i, file in enumerate(files):
    #test_size = float(file.split("_")[1].replace(".dat", ""))
    plt.subplot(121)
    fn = file
    fidr = open(fn, 'rb')
    obj = pickle.load(fidr)
    fidr.close()
    plt.plot(obj.df_results['iter'], obj.df_results['score'].abs(), label = " Run {}".format(i))
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood\n $\mathcal{L}( f, \mathcal{D})$ ")
    plt.title("(a) Progress of MCMC")

    if i == 0:
        score = obj.mean_score.copy()
        score.set_index(obj.sample_id, inplace = True)
        score.rename(columns = {'score_mean':"run {}".format(i)}, inplace = True)
    else:
        sc = obj.mean_score.copy()
        sc.set_index(obj.sample_id, inplace=True)
        nm = "run {}".format(i)
        score[nm] = sc['score_mean']

plt.subplot(122)
sig_flg = obj.df.set_index('sample_id')
# sig_flg.loc[sig_flg['signal'].isna(), 'signal'] = 0
# score['True_Score'] = sig_flg['signal'].astype('float')

cmap = sns.diverging_palette(230, 20, as_cmap=True)
corr = score.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot= True)
plt.title("(b) Correlation of Scores \n from multiple runs")
plt.tight_layout()




