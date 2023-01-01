import os
import matplotlib.pyplot as plt
import pandas as pd
import outlier_detector
import pickle
from utils import get_scores
import shap


files = [ r"ca_wu\ca_wu_od_1253.dat",  r"ca_wu\ca_wu_od_2130.dat",  r"ca_wu\ca_wu_od_5523.dat", r"ca_wu\ca_wu_od_8891.dat"]
av_score = 0
for i, fn in enumerate(files):
    fidr = open(fn, 'rb')
    obj = pickle.load(fidr)
    fidr.close()
    scores = get_scores(obj, burn_in=150)
    #scores = obj.mean_score
    ss = scores.sort_values(by='sample_id')
    av_score = av_score + ss['score_mean'].values

av_score = av_score/4.0
ss['score_mean'] = av_score
df = obj.df.copy()
scores = ss

df = df.merge(scores, how = 'left', on = 'sample_id')



fig, ax = plt.subplots()
ax2 = ax.twinx()
series = df['score_mean'].copy()
ax.hist(series, bins=50, label='FDF')
ax2.hist(
    series, cumulative=1, histtype='step', bins=50, color='tab:red', density=True)
ax2.hist(
    series, cumulative=1, bins=50, alpha = 0.1, color='tab:red', density=True, label='CDF')
plt.xlim([0,1])
ax.set_xlabel("Outlier Score")

ax.set_ylabel("Frequency")
ax2.set_ylabel("CDF")
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)


features = obj.features + [obj.target]
target = 'score_mean'

ml_hyperparamters = {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "colsample_bytree": 0.8,
                "learning_rate": 0.08,
                "max_depth": 7,
                "alpha": 0.001,
                "n_estimators": 500,
                "subsample": 0.9,
                "reg_lambda": 11,
                "min_child_weight": 10,
                "gamma": 5e-4,
                "seed": 123,
            }
classification_params = {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "colsample_bytree": 0.8,
                "learning_rate": 0.08,
                "max_depth": 7,
                "alpha": 0.001,
                "n_estimators": 500,
                "subsample": 0.9,
                "reg_lambda": 11,
                "min_child_weight": 10,
                "gamma": 5e-4,
                "seed": 123,
            }



smodel = outlier_detector.ScoreModel(df = df,
                            features = features,
                            target = target,
                            seed=123,
                            test_fraction=0.2,
                            estimator=None,
                            ml_hyperparamters=ml_hyperparamters
                            )
smodel.train()
cmodel, df, accurecy, valid_df = smodel.classify(classification_params)

agreement_df = valid_df[valid_df['y_true']==valid_df['y_hat']]
disagreement_df = valid_df[valid_df['y_true']!=valid_df['y_hat']]
import seaborn
import matplotlib.pyplot as plt
# declaring data


# define Seaborn color palette to use
palette_color = seaborn.color_palette('Set2')
data = [(agreement_df['y_true']==0).sum(),
        (agreement_df['y_true']==1).sum()]
keys = ['Outliers', 'Signal']

plt.subplot(121)
# plotting data on chart
plt.pie(data, labels=keys, colors=palette_color, autopct='%.0f%%')
plt.title(" {} % of the holdout dataset\ncorrectly classified".format(round(100*accurecy, 2)))
# displaying chart
plt.show()

palette_color = seaborn.color_palette('Set2')
data = [(disagreement_df['y_true']==0).sum(),
        (disagreement_df['y_true']==1).sum()]
keys = ['Outliers', 'Signal']

plt.subplot(122)
# plotting data on chart
plt.pie(data, labels=keys, colors=palette_color, autopct='%.0f%%')
plt.title(" {} % of the holdout dataset\nincorrectly classified".format(round(100*(1-accurecy), 2)))
# displaying chart
plt.show()


## Importance plot
import numpy as np
importances = cmodel.feature_importances_
indices = np.argsort(importances)[-10:]
features = df.columns
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='g', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


###
# Global
df11 = df[df['score_mean']<1]
X100 = df11.sample(frac=0.5).copy()
X100 = X100[features]
explainer = shap.Explainer(cmodel, X100)
shap_values = explainer(X100)
shap.summary_plot(
    shap_values, X100, max_display=15, show=False, color_bar=False
)
cb = plt.colorbar(shrink=0.2, label="Feature Value")
cb.ax.tick_params(labelsize=500)
cb.set_ticks([])
cb.ax.text(
    0.5, -0.01, "Low", transform=cb.ax.transAxes, va="top", ha="center"
)
cb.ax.text(
    0.5,
    1.01,
    "High",
    transform=cb.ax.transAxes,
    va="bottom",
    ha="center",
)
plt.gca().figure.suptitle("Global SHAP Values", fontsize=14)
plt.tight_layout()
pdf.savefig()
plt.close()

for feat in features:
    try:
        fig = plt.figure()
        cax = fig.gca()
        shap.plots.scatter(
            shap_values[:, feat],
            color=shap_values,
            ax=cax,
            hist=False,
            show=False,
        )
        fig.suptitle("SHAP Values for {}".format(feat), fontsize=14)
        vrange = np.abs(
            X100[feat].quantile(0.90) / X100[feat].quantile(0.1)
        )
        if vrange > 0:
            pvals = X100[feat].values
            pvals = pvals[pvals != -np.inf]
            pvals = pvals[pvals != np.inf]
            max_limit = np.nanmax(pvals)
            min_limit = np.nanmin(pvals)
            plt.xlim([min_limit, max_limit])
            if (max_limit > 0) & (min_limit > 0):
                plt.xscale("log")
        parname = fig.axes[-1].get_ylabel()
        fig.axes[-1].remove()
        PCM = cax.get_children()[2]
        cb = plt.colorbar(PCM, ax=cax, shrink=1.0, label=parname)
        cb.ax.tick_params(labelsize=10)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    except:
        print("Issues with SHAP values for {}".format(feat))

# per HUC2
hru2s = np.sort(df["HUC2"].unique())
for h in hru2s:
    print("Explaining HUC2: {}".format(h))
    XX = df[df["HUC2"] == h]
    XX = XX[features]
    explainer = shap.Explainer(estimator, XX)
    shap_values = explainer(XX)
    shap.summary_plot(
        shap_values, XX, max_display=15, color_bar=False, show=False
    )
    cb = plt.colorbar(shrink=0.2, label="Feature Value")
    cb.ax.tick_params(labelsize=12)
    cb.set_ticks([])
    cb.ax.text(
        0.5,
        -0.01,
        "Low",
        transform=cb.ax.transAxes,
        va="top",
        ha="center",
    )
    cb.ax.text(
        0.5,
        1.01,
        "High",
        transform=cb.ax.transAxes,
        va="bottom",
        ha="center",
    )
    fig = plt.gca().figure
    fig.suptitle("SHAP Values for HUC2 = {}".format(h), fontsize=12)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

data = 1