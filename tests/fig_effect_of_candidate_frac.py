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


figures_folder = r"results\figures"
data_folder =  r"results\mov_frac"

files = os.listdir(data_folder)

width_in = 7
height_in = 6
fig = plt.figure(figsize=(width_in, height_in))

jj = 0
frrr = []
success_rate = []
burn_in = []
stationary_std = []
stationary_likelihood = []
average_acceptance_rate = []
for i, file in enumerate(files):
    # if file in ['run_0.07.dat', 'run_0.005.dat', 'run_0.05.dat']:
    #     continue

    test_size = float(file.split("_")[1].replace(".dat", ""))
    frrr.append(str(test_size))

    fn = os.path.join(data_folder, file)
    fidr = open(fn, 'rb')
    obj = pickle.load(fidr)
    fidr.close()
    # plt.plot(obj.df_results['iter'],obj.acceptance_rate, label = " fraction {}".format(test_size))
    # plt.legend()
    # plt.xlabel("Iteration")
    # plt.ylabel("Mean Square Error")
    # plt.title("(a) Progress of MCMC")

    import skgstat as skg

    phi = -1*np.array(get_burin_period(obj, 20))
    x = np.arange(len(phi))
    field = np.abs(phi)
    #field = np.abs(obj.signal_iter_score)

    # V = skg.Variogram(coordinates=x, values=field)
    # brin, sill_value = V.cof[0], V.cof[1]
    import gstools as gs


    m = gs.Stable(dim=1)
    m.fit_variogram(x, max(field) - field, nugget=False)
    len_scale = int(m.len_scale) + 2
    stddd = np.std(phi[len_scale:])
    burn_in.append(len_scale)
    stationary_std.append(stddd**2.0)
    stationary_likelihood.append(np.mean(phi[len_scale:]))
    average_acceptance_rate.append(np.mean(obj.acceptance_rate[len_scale:]))

    suc_rate = get_success_rate(obj, 'signal')
    success_rate.append(suc_rate)
    if jj == 0:
        score = obj.mean_score.copy()
        score.set_index(obj.sample_id, inplace = True)
        score.rename(columns = {'score_mean':"run_{}".format(i)}, inplace = True)
    else:
        sc = obj.mean_score.copy()
        sc.set_index(obj.sample_id, inplace=True)
        nm = "run_{}".format(jj)
        score[nm] = sc['score_mean']
    jj = jj + 1


results = pd.DataFrame()
results['Update Step Size'] = frrr
results['Length of Burn-in Period'] = burn_in
results['Likelihood STD'] = stationary_std
results['Likelihood Mean'] = stationary_likelihood
results['Success Rate'] = success_rate

figure, axes = plt.subplots(2, 2, sharex=True, figsize=(10,5))
#figure.suptitle('MCMC Hyperparameters')

colors = sns.color_palette("husl", 9)
barcolor = sns.color_palette("RdBu", 10)

axes[0,0].set_title('(a)')
sns.barplot(ax =axes[0,0],
    data=results,
    x="Update Step Size", y='Length of Burn-in Period',color=barcolor[-1])
sns.lineplot(ax =axes[0,0],
    data=results,
    x="Update Step Size", y='Length of Burn-in Period',

             markers=True, dashes=False, marker = 'o',
             color = colors[0],
        markerfacecolor = 'k'
)
axes[0,0].set_xlabel(r"Update Step Size ($\beta_s$)")


#axes[0,1].set_title('second chart with no data')
axes[0,1].set_title('(b)')
sns.barplot(ax =axes[0,1],
    data=results,
    x="Update Step Size", y="Likelihood Mean",color=barcolor[0])
sns.lineplot(ax = axes[0,1],
    data=results,
    x="Update Step Size", y="Likelihood Mean",marker = 's',

             markers=True, dashes=False,
             color = colors[1],
markerfacecolor = 'k'
)
axes[0,1].set_xlabel(r"Update Step Size ($\beta_s$)")
axes[0,1].set_ylabel("Likelihood Mean \n $E[\mathcal{L}( f, \mathcal{D})]$")

axes[1,0].set_title('(c)')
sns.barplot(ax =axes[1,0],
    data=results,
    x="Update Step Size", y="Likelihood STD",color=barcolor[-3])

sns.lineplot(ax = axes[1,0],
    data=results,
    x="Update Step Size", y="Likelihood STD",marker = 'x',

             markers=True, dashes=False,
            color = colors[2]
             ,markeredgecolor = 'k'
)
axes[1,0].set_xlabel(r"Update Step Size ($\beta_s$)")
axes[1,0].set_ylabel("Likelihood Variance \n $Var[\mathcal{L}( f, \mathcal{D})]$")


axes[1,1].set_title('(d)')
sns.barplot(ax =axes[1,1],
    data=results,
    x="Update Step Size", y="Success Rate",color=barcolor[2])
sns.lineplot(ax = axes[1,1],
    data=results,
    x="Update Step Size", y="Success Rate",marker = '*',

             markers=True, dashes=False,
            color = colors[3],markerfacecolor = 'k'
)
axes[1,1].set_xlabel(r"Update Step Size ($\beta_s$)")

plt.tight_layout()
plt.show()
cc = 1





