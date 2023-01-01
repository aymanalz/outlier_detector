import os, sys
import matplotlib.pyplot as plt
import pickle

import pandas as pd
import seaborn as sns
import numpy as np


def get_success_rate(obj, true_flg_col):
    """

    :param obj:
    :param true_flg_col: the pandas column name with outliers flag
    :return:
    """
    df = obj.df.copy()
    df = df[[obj.sample_id, true_flg_col]]
    df.loc[df[true_flg_col].isna(), true_flg_col] = 0
    df = df.merge(obj.mean_score, how='left', on=obj.sample_id)
    df.loc[df['score_mean'] >= 0.5, 'score_mean'] = 1
    df.loc[df['score_mean'] < 0.5, 'score_mean'] = 0
    success_rate = 100*(df[true_flg_col] == df['score_mean']).sum()/len(df)
    return success_rate

def get_burin_period(obj, window):
    import copy
    phi = copy.deepcopy(obj.signal_iter_score)
    phi_mean = []
    phi_std = []
    for i, v in enumerate(phi):
        end = i+1

        if i < window:
            start = 0
        else:
            start = i - window
        phi_mean.append(np.mean(phi[start:end]))
        phi_std.append(np.std(phi[start:end]))

    return phi_mean

files = ['random_walk.dat', 'quantile.dat', 'mse.dat']
outlier_frac = ['Random Walk','Error Quantile', 'Squared Error']
success_rate = []
data_folder =  r"results\smaplers"
width_in = 7
height_in = 3
fig = plt.figure(figsize=(width_in, height_in))
metrics = {}
for i, file in enumerate(files):
    test_size =outlier_frac[i]
    plt.subplot(121)
    fn = os.path.join(data_folder, file)
    fidr = open(fn, 'rb')
    obj = pickle.load(fidr)
    nsamples = len(obj.df)
    fidr.close()
    plt.plot(obj.df_results['iter'], obj.df_results['score'].abs()/nsamples, label = " {}".format(test_size))
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood\n $\mathcal{L}( f, \mathcal{D})$ ")
    plt.title("(a) Progress of MCMC")

    import skgstat as skg


    curr_results = []
    phi = -1 * np.array(get_burin_period(obj, 20))
    x = np.arange(len(phi))
    field = np.abs(phi)
    # field = np.abs(obj.signal_iter_score)

    # V = skg.Variogram(coordinates=x, values=field)
    # brin, sill_value = V.cof[0], V.cof[1]
    import gstools as gs

    m = gs.Spherical(dim=1)
    m.fit_variogram(x, max(field) - field, nugget=False)
    len_scale = int(m.len_scale) + 2
    stddd = np.std(phi[len_scale:])
    #burn_in.append(len_scale)
    #stationary_std.append(stddd ** 2.0)
    #stationary_likelihood.append()
    #average_acceptance_rate.append(np.mean(obj.acceptance_rate[len_scale:]))

    curr_results.append(len_scale)
    curr_results.append(stddd**2.0)
    curr_results.append(np.mean(phi[len_scale:]))
    curr_results.append(np.mean(obj.acceptance_rate[len_scale:]))
    metrics[test_size] = curr_results



    suc_rate = get_success_rate(obj, 'signal')
    success_rate.append(suc_rate)

    score = obj.mean_score.copy()
    score.set_index(obj.sample_id, inplace = True)
    score.rename(columns = {'score_mean':"run".format(i)}, inplace = True)


    sig_flg = obj.df.set_index('id')
    sig_flg.loc[sig_flg['signal'].isna(), 'signal'] = 0
    score['True_Score'] = sig_flg['signal'].astype('float')
    score['bin_flg'] = 0
    score.loc[score['run']>=0.5, 'bin_flg'] = 1
    # sr = int(10000*(score['True_Score'] == score['bin_flg']).sum())/len(score)/100
    # success_rate.append(sr)

plt.subplot(122)
plt.title("(b) Success Rate for MCMC\n Proposal Functions")
Ssc_df = pd.DataFrame(columns=['Method', 'success_rate'])
Ssc_df['Method'] = outlier_frac
Ssc_df['success_rate'] = success_rate
ax = sns.barplot(data = Ssc_df, x='Method', y = 'success_rate')
ax.set_xlabel("Proposal Function Type")
ax.set_ylabel("Success Rate (%)")
for i in ax.containers:
    ax.bar_label(i,)
df = pd.DataFrame(metrics, index = ['Burn_in Period', 'Likelihood Variance', 'Likelihood Mean', 'Acceptance Rate'])
df.to_csv("sampler_metrics.csv")
cc = 1