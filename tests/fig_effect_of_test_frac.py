import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

figures_folder = r"results\figures"
data_folder =  r"results\test_fact"

files = os.listdir(data_folder)

width_in = 7
height_in = 6
fig = plt.figure(figsize=(width_in, height_in))


for i, file in enumerate(files):
    test_size = float(file.split("_")[1].replace(".dat", ""))
    plt.subplot(221)
    fn = os.path.join(data_folder, file)
    fidr = open(fn, 'rb')
    obj = pickle.load(fidr)
    fidr.close()
    plt.plot(obj.df_results['iter'], obj.df_results['score'].abs(), label = " fraction {}".format(test_size))
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Mean Square Error")
    plt.title("(a) Progress of MCMC")

    if i == 0:
        score = obj.mean_score.copy()
        score.set_index(obj.sample_id, inplace = True)
        score.rename(columns = {'score_mean':"run_{}".format(i)}, inplace = True)
    else:
        sc = obj.mean_score.copy()
        sc.set_index(obj.sample_id, inplace=True)
        nm = "run_{}".format(i)
        score[nm] = sc['score_mean']

plt.subplot(222)
sig_flg = obj.df.set_index('id')
sig_flg.loc[sig_flg['signal'].isna(), 'signal'] = 0
score['True_Score'] = sig_flg['signal'].astype('float')

cmap = sns.diverging_palette(230, 20, as_cmap=True)
corr = score.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot= True)
plt.title("(b) Correlation of Scores \n from multiple runs")

plt.subplot(223)
for c in score.columns:
    if "run_" in c:
        score[c].plot.kde(bw_method=0.05, label = c)
        plt.xlim([0,1])
plt.xlabel("Outlier Scores")
plt.ylabel("frequency")
plt.title("(c) Score Density")


plt.subplot(224)
xy_data = obj.df.set_index('id')
xy_data['score'] = score['run_1']
plt.scatter(xy_data['x1'], xy_data['y'], c = xy_data['score'], s = 5, cmap = 'jet')
plt.colorbar()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title("(d) Outlier Scores for Run 1")

plt.tight_layout()
plt.show()
cc = 1





