import os, sys
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np
import pandas as pd
files = ['noise_ratio_25.dat', 'noise_ratio_50.dat', 'noise_ratio_100.dat',
         'D6_noise_ratio_0_25.dat', 'D6_noise_ratio_0_50.dat', 'D6_noise_ratio_0_100.dat']
outlier_frac = [0.25, 0.5,1.0, 0.25, 0.5, 1.0]
success_rate = []
data_folder =  r"results\noise_sig_ratio"
width_in = 12
height_in = 6
def annotate_axes(ax, text, fontsize=18):
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="darkgrey")
#fig = plt.figure(figsize=(width_in, height_in))
fig, axd = plt.subplot_mosaic([['upper left', 'right'],
                               ['lower left', 'right']],
                              figsize=(5.5, 3.5))#, layout="constrained"
# for k in axd:
#     annotate_axes(axd[k], f'axd["{k}"]', fontsize=14)
#fig.suptitle('Effect of Outliers-to-Signal Ratio on the Algorithm Performance ')

for i, file in enumerate(files):
    test_size =outlier_frac[i]
    if i < 3:
        #plt.subplot(221)
        fn = os.path.join(data_folder, file)
        fidr = open(fn, 'rb')
        obj = pickle.load(fidr)
        nsamples = len(obj.df)
        fidr.close()
        ax = axd['upper left']
        ax.plot(obj.df_results['iter'], obj.df_results['score'].abs(), label = "ratio = {}".format(test_size))

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log-Likelihood\n $\mathcal{L}( f, \mathcal{D})$ ")
        ax.set_title("(a) MCMC Progress for \n the 1D Case")
        ax.legend()
        #ax.legend(bbox_to_anchor=(1.02, 0.55), loc='upper right', borderaxespad=0)
    else:
        #plt.subplot(222)
        fn = os.path.join(data_folder, file)
        fidr = open(fn, 'rb')
        obj = pickle.load(fidr)
        nsamples = len(obj.df)
        fidr.close()
        ax = axd['lower left']
        ax.plot(obj.df_results['iter'], obj.df_results['score'].abs() , label="ratio = {}".format(test_size))

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log-Likelihood\n $\mathcal{L}( f, \mathcal{D})$ ")
        ax.set_title("(b) MCMC Progress for \n the 6D Case")
        ax.legend()


    score = obj.mean_score.copy()
    score.set_index(obj.sample_id, inplace=True)
    score.rename(columns={'score_mean': "run".format(i)}, inplace=True)

    sig_flg = obj.df.set_index(obj.sample_id)
    sig_flg.loc[sig_flg['signal'].isna(), 'signal'] = 0
    score['True_Score'] = sig_flg['signal'].astype('float')
    score['bin_flg'] = 0
    score.loc[score['run'] >= 0.5, 'bin_flg'] = 1
    sr = int(10000 * (score['True_Score'] == score['bin_flg']).sum()) / len(score) / 100
    success_rate.append(sr)

ax =  axd['right']
Ssc_df = pd.DataFrame(columns=['Method', 'success_rate'])
Ssc_df['Method'] = outlier_frac
Ssc_df['success_rate'] = success_rate
Ssc_df['Dataset'] = '6D'
Ssc_df.loc[Ssc_df.index<3, 'Dataset'] = '1D'
Ssc_df['success_rate'] = Ssc_df['success_rate'].round(1)
ax1 = sns.barplot(data=Ssc_df, x='Method', y='success_rate', hue="Dataset", ax =ax)
for i in ax1.containers:
    ax1.bar_label(i, )
ax.set_xlabel([0,150])
ax.set_title("(c) Success Rate")
#plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
ax.set_xlabel("Outlier to Signal Ratio")
ax.set_ylabel("Success Rate")
plt.tight_layout()
vv = 1

# ======
if 0:
    files = ['D6_noise_ratio_0_100.dat', 'D6_noise_ratio_enhanced_100.dat', 'D6_noise_ratio_mse_enhanced_100.dat',
             'D6_noise_ratio_quantile2_enhanced_100.dat']
    outlier_frac = ['1D', '6D-QE', '6D-SE', '6D-QE2']
    success_rate = []
    data_folder =  r"results\noise_sig_ratio"
    width_in = 7
    height_in = 6
    def annotate_axes(ax, text, fontsize=18):
        ax.text(0.5, 0.5, text, transform=ax.transAxes,
                ha="center", va="center", fontsize=fontsize, color="darkgrey")
    #fig = plt.figure(figsize=(width_in, height_in))
    fig, axd = plt.subplot_mosaic([['upper left', 'right'],
                                   ['lower left', 'right']],
                                  figsize=(5.5, 3.5))#, layout="constrained"
    # for k in axd:
    #     annotate_axes(axd[k], f'axd["{k}"]', fontsize=14)
    fig.suptitle('Effect of Outliers to Signal Ratio on the Algorithm Performance ')

    for i, file in enumerate(files):
        test_size =outlier_frac[i]
        if i < 3:
            #plt.subplot(221)
            fn = os.path.join(data_folder, file)
            fidr = open(fn, 'rb')
            obj = pickle.load(fidr)
            nsamples = len(obj.df)
            fidr.close()
            ax = axd['upper left']
            ax.plot(obj.df_results['iter'], obj.df_results['score'].abs()/nsamples, label = "ratio = {}".format(test_size))
            plt.legend()
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Log-Likelihood")
            ax.set_title("(a) Convergence of the 1D Case")
            ax.legend()
            #ax.legend(bbox_to_anchor=(1.02, 0.55), loc='upper right', borderaxespad=0)
        else:
            #plt.subplot(222)
            fn = os.path.join(data_folder, file)
            fidr = open(fn, 'rb')
            obj = pickle.load(fidr)
            nsamples = len(obj.df)
            fidr.close()
            ax = axd['lower left']
            ax.plot(obj.df_results['iter'], obj.df_results['score'].abs() / nsamples, label="ratio = {}".format(test_size))

            ax.set_xlabel("Iteration")
            ax.set_ylabel("Log-Likelihood")
            ax.set_title("(b) Convergence for the 6D Case")
            ax.legend()



        score = obj.mean_score.copy()
        score.set_index(obj.sample_id, inplace=True)
        score.rename(columns={'score_mean': "run".format(i)}, inplace=True)

        sig_flg = obj.df.set_index(obj.sample_id)
        sig_flg.loc[sig_flg['signal'].isna(), 'signal'] = 0
        score['True_Score'] = sig_flg['signal'].astype('float')
        score['bin_flg'] = 0
        score.loc[score['run'] >= 0.5, 'bin_flg'] = 1
        sr = int(10000 * (score['True_Score'] == score['bin_flg']).sum()) / len(score) / 100
        success_rate.append(sr)

    ax =  axd['right']
    Ssc_df = pd.DataFrame(columns=['Method', 'success_rate'])
    Ssc_df['Method'] = outlier_frac
    Ssc_df['success_rate'] = success_rate
    Ssc_df['Dataset'] = '6D'
    Ssc_df.loc[Ssc_df.index<3, 'Dataset'] = '1D'
    Ssc_df['success_rate'] = Ssc_df['success_rate'].round(1)
    ax1 = sns.barplot(data=Ssc_df, x='Method', y='success_rate', hue="Dataset", ax =ax)
    for i in ax1.containers:
        ax1.bar_label(i, )
    ax.set_xlabel([0,150])
    ax.set_title("(c) Success Rate")
    #plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
    ax.set_xlabel("Outlier to Signal Ratio")
    ax.set_ylabel("Success Rate")
    plt.tight_layout()

    cc = 1