import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

figures_folder = r"results\figures"
data_folder =  r"results\multi_start"

files = os.listdir(data_folder)


figure, axes = plt.subplots(2, 2, figsize=(12,7))
success_rate = []
for i, file in enumerate(files):
    #plt.subplot(221)
    fn = os.path.join(data_folder, file)
    fidr = open(fn, 'rb')
    obj = pickle.load(fidr)
    fidr.close()
    obj.df_results['score'] = obj.df_results['score'].abs()
    ax = axes[0, 0]
    #plt.plot(obj.df_results['iter'], obj.df_results['score'].abs(), label = "Run {}".format(i))
    sns.lineplot(ax = ax, data=obj.df_results, x="iter", y="score",  label = "Run {}".format(i))
    ax.legend()
    ax.set_xlabel("Iteration")
    ax.legend()
    ax.set_ylabel("Log-Likelihood\n $\mathcal{L}( f, \mathcal{D})$ ")
    ax.set_title("(a) Progress of MCMC")
    success_rate.append(get_success_rate(obj, 'signal'))

    if i == 0:
        import utils
        score = utils.get_scores(obj, 200)
        #score = obj.mean_score.copy()
        score.set_index(obj.sample_id, inplace = True)
        score.rename(columns = {'score_mean':"run {}".format(i)}, inplace = True)
    else:
        import utils
        sc = utils.get_scores(obj, 200)
        #sc = obj.mean_score.copy()
        sc.set_index(obj.sample_id, inplace=True)
        nm = "run {}".format(i)
        score[nm] = sc['score_mean']

ax = axes[0, 1]
sig_flg = obj.df.set_index('id')
sig_flg.loc[sig_flg['signal'].isna(), 'signal'] = 0
score['True Score'] = sig_flg['signal'].astype('float')

cmap = sns.diverging_palette(230, 20, as_cmap=True)
corr = score.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap( corr, square=False, linewidths=.5, cbar_kws={"shrink": .5}, annot= True, ax = ax)
ax.set_title("(b) Correlation of Scores \n from Multiple Runs")

ax = axes[1, 0]
for c in score.columns:
    if "run" in c:
        score[c].plot.kde(bw_method=0.02, label = c, ax = ax)
        ax.set_xlim([0,1])
ax.legend()
ax.set_xlabel("Outlier Scores")
ax.set_ylabel("Frequency")
ax.set_title("(c) Score Density")


ax = axes[1, 1]
xy_data = obj.df.set_index('id')
xy_data['score'] = (score['run 0'] + score['run 1'] + score['run 2'])/3
#plt.scatter(xy_data['x1'], xy_data['y'], c = xy_data['score'], s = 5, cmap = 'jet')
sns.scatterplot(data=xy_data, x="x1", y="y", hue="score", ax = ax, palette='Spectral',  s=12)
ax.set_xlim([xy_data['x1'].min(), xy_data['x1'].max()])
norm = plt.Normalize(xy_data['score'].min()-0.001, xy_data['score'].max()+0.001)
sm = plt.cm.ScalarMappable(cmap="Spectral", norm=norm)
sm.set_array([])

# Remove the legend and add a colorbar
ax.get_legend().remove()
ax.figure.colorbar(sm)
ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")
ax.set_title("(d) Outlier Average Scores for\n Runs 1, 2, and 3")

plt.tight_layout()

cc = 1





