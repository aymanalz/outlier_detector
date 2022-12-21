import os, sys
import matplotlib.pyplot as plt
import pickle

import pandas as pd
import seaborn as sns
import numpy as np
files = ['random_walk_proposal.dat', 'quantile_proposal.dat', 'mse_proposal.dat']
outlier_frac = ['Random Walk','Quantile', 'Square Error']
success_rate = []
data_folder =  r"results\smaplers"
width_in = 7
height_in = 6
fig = plt.figure(figsize=(width_in, height_in))
for i, file in enumerate(files):
    test_size =outlier_frac[i]
    plt.subplot(221)
    fn = os.path.join(data_folder, file)
    fidr = open(fn, 'rb')
    obj = pickle.load(fidr)
    nsamples = len(obj.df)
    fidr.close()
    plt.plot(obj.df_results['iter'], obj.df_results['score'].abs()/nsamples, label = " {}".format(test_size))
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.title("(a) Progress of MCMC")


    score = obj.mean_score.copy()
    score.set_index(obj.sample_id, inplace = True)
    score.rename(columns = {'score_mean':"run".format(i)}, inplace = True)


    sig_flg = obj.df.set_index('id')
    sig_flg.loc[sig_flg['signal'].isna(), 'signal'] = 0
    score['True_Score'] = sig_flg['signal'].astype('float')
    score['bin_flg'] = 0
    score.loc[score['run']>=0.5, 'bin_flg'] = 1
    sr = int(10000*(score['True_Score'] == score['bin_flg']).sum())/len(score)/100
    success_rate.append(sr)

plt.subplot(222)
Ssc_df = pd.DataFrame(columns=['Method', 'success_rate'])
Ssc_df['Method'] = outlier_frac
Ssc_df['success_rate'] = success_rate
ax = sns.barplot(data = Ssc_df, x='Method', y = 'success_rate')
for i in ax.containers:
    ax.bar_label(i,)

cc = 1