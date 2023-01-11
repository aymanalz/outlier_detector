import os
import matplotlib.pyplot as plt
import pandas as pd
import outlier_detector
import pickle
from utils import get_scores
import shap
import xgboost as xgb
from sklearn.metrics import r2_score
import numpy as np


files = [ r"ca_wu\ca_wu_od_1253.dat",  r"ca_wu\ca_wu_od_2130.dat",  r"ca_wu\ca_wu_od_5523.dat",
          r"ca_wu\ca_wu_od_8891.dat", r"ca_wu\ca_wu_od_52347.dat"]
av_score = 0
for i, fn in enumerate(files):
    fidr = open(fn, 'rb')
    obj = pickle.load(fidr)
    fidr.close()
    scores = get_scores(obj, burn_in=150)
    #scores = obj.mean_score
    ss = scores.sort_values(by='sample_id')
    av_score = av_score + ss['score_mean'].values

av_score = av_score/5.0
ss['score_mean'] = av_score
df = obj.df.copy()
scores = ss
df = df.merge(scores, how = 'left', on = 'sample_id')

### predictive model
plt.subplot(131)
features = obj.features
target = [obj.target]
params = {
    "objective": "reg:squarederror",
    "base_score": 0.5,
    "booster": "gbtree",
    "callbacks": None,
    "colsample_bylevel": 1,
    "colsample_bynode": 1,
    "colsample_bytree": 0.867202783570103,
    "early_stopping_rounds": None,
    "enable_categorical": False,
    "eval_metric": None,
    "gamma": 0,
    "grow_policy": "depthwise",
    "importance_type": None,
    "interaction_constraints": "",
    "learning_rate": 0.12128959372061261,
    "max_bin": 256,
    "max_cat_to_onehot": 4,
    "max_delta_step": 0,
    "max_depth": 11,
    "max_leaves": 0,
    "min_child_weight": 4,
    "monotone_constraints": "()",
    "n_estimators": 300,
    "n_jobs": -1,
    "num_parallel_tree": 1,
    "predictor": "auto",
    "random_state": 5751,
    "reg_alpha": 10.0,
    "reg_lambda": 10.0,
    "sampling_method": "uniform",
    "scale_pos_weight": 50.0,
    "subsample": 0.867555264259934,
    "tree_method": "hist",
    "validate_parameters": 1,
    "verbosity": 0,
}

df_ = df.copy()
df_ = df_.sample(frac=1, random_state=785).reset_index(drop=True)
train_df = df_.sample(frac=(1 - 0.3), random_state=785)
test_df = df_.drop(index=train_df.index)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
gb = xgb.XGBRegressor(**params)
gb.set_params(random_state= 785)
gb.set_params(seed= 785)
gb.fit(train_df[features], train_df[target])

y_hat = gb.predict(test_df[features])
plt.scatter(y_hat, test_df[target], s = 20, edgecolors = 'w', c = 'g', alpha = 0.5)
miv = min(np.min(y_hat), np.min(test_df[target].values))
mxv = max(np.max(y_hat), np.max(test_df[target].values))
plt.plot([miv, mxv], [miv, mxv], label = "1-1 Line", c = 'r')
r22 = int(1000 * r2_score(y_hat, test_df[target]))/1000
plt.title("(a) Raw Data \n $R^2$ = {}".format(r22))
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Predicted Per Capita\n Water Use [GPCD]")
plt.ylabel("True Per Capita\n Water Use [GPCD]")
plt.legend()



# drop outliers
plt.subplot(132)
df_ = df.copy()
df_ = df_[df_['score_mean']>=0.3]
df_ = df_.sample(frac=1, random_state=785).reset_index(drop=True)
train_df = df_.sample(frac=(1 - 0.3), random_state=785)
test_df = df_.drop(index=train_df.index)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
gb = xgb.XGBRegressor(**params)
gb.set_params(random_state= 785)
gb.set_params(seed= 785)
gb.fit(train_df[features], train_df[target])


y_hat = gb.predict(test_df[features])
plt.scatter(y_hat, test_df[target], s = 20, edgecolors = 'w', c = 'b', alpha = 0.5)
miv = min(np.min(y_hat), np.min(test_df[target].values))
mxv = max(np.max(y_hat), np.max(test_df[target].values))
plt.plot([miv, mxv], [miv, mxv], label = "1-1 Line", c = 'r')
r22 = int(1000 * r2_score(y_hat, test_df[target]))/1000
plt.title("(b) Outliers Dropped \n $R^2$ = {}".format(r22))
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Predicted Per Capita\n Water Use [GPCD]")
plt.ylabel("True Per Capita\n Water Use [GPCD]")
plt.legend()

# weight outliers
plt.subplot(133)
df_ = df.copy()
df_ = df_.sample(frac=1, random_state=785).reset_index(drop=True)
train_df = df_.sample(frac=(1 - 0.3), random_state=785)
test_df = df_.drop(index=train_df.index)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
w = np.exp(10 * train_df['score_mean'].values)

gb = xgb.XGBRegressor(**params)
gb.set_params(random_state= 785)
gb.set_params(seed= 785)
gb.fit(train_df[features], train_df[target], sample_weight=w)

y_hat = gb.predict(test_df[features])

w = np.exp(10 * test_df['score_mean'].values)
r22 = int(1000 * r2_score(y_hat, test_df[target], sample_weight=w))/1000
y_hat = gb.predict(test_df[features])
im = plt.scatter(y_hat, test_df[target], s = 20, edgecolors = 'w', c = test_df['score_mean'].values, alpha = 0.7, cmap = 'jet_r')
miv = min(np.min(y_hat), np.min(test_df[target].values))
mxv = max(np.max(y_hat), np.max(test_df[target].values))
plt.plot([miv, mxv], [miv, mxv], label = "1-1 Line", c = 'r')

plt.title("(c) Outliers Weighted \n $R^2$ = {}".format(r22))
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Predicted Per Capita\n Water Use [GPCD]")
plt.ylabel("True Per Capita\n Water Use [GPCD]")

plt.legend()

ax = plt.gca()
from mpl_toolkits.axes_grid1 import make_axes_locatable

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax)




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

cmodel, df, accurecy, valid_df = smodel.classify(classification_params)
rmodel, df_reg, accurecy_reg, valid_df_reg = smodel.train(ml_hyperparamters)


### confusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
plt.figure()
plt.subplot(121)
cf_matrix = confusion_matrix(valid_df['y_true'], valid_df['y_hat'])
group_names = ['True Outliers','False Signals','False Outliers','True Signals']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
ax2 = plt.gca()
ax = sns.heatmap( cf_matrix, annot=labels, fmt="", cmap='Blues', ax=ax2)
Accuracy = int(1000 * np.diag(cf_matrix).sum()/cf_matrix.sum())/1000
ax.set_xlabel("Predicted Score")
plt.title("(a) Score Classifier Model\n $Accuracy$ = {}".format(Accuracy))
ax.set_ylabel("True Score")


# regression eval
plt.subplot(122)
y_hat = valid_df_reg['y_hat']
y_true = valid_df_reg['y_true']
plt.scatter(y_hat, y_true, s = 20, edgecolors = 'w', c = 'b', alpha = 0.5)
miv = min(np.min(y_hat), np.min(y_true))
mxv = max(np.max(y_hat), np.max(y_true))
plt.plot([miv, mxv], [miv, mxv], label = "1-1 Line", c = 'r')
r22 = int(1000 * r2_score(y_hat, y_true))/1000
plt.title("(b) Score Regression Model  \n $R^2$ = {}".format(r22))
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Predicted Score")
plt.ylabel("True Score")
plt.legend()


if 0:
    agreement_df = valid_df[valid_df['y_true'] == valid_df['y_hat']]
    disagreement_df = valid_df[valid_df['y_true'] != valid_df['y_hat']]
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
plt.figure()



###
# Global
plt.subplot(121)
df11 = df[df['score_mean']<1]
X100 = df11.sample(frac=0.5).copy()
X100 = X100[features]
explainer = shap.Explainer(rmodel, X100)
shap_values = explainer(X100)
shap.summary_plot(
    shap_values, X100, max_display=10, show=False, color_bar=False
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
plt.title("(a) Global SHAP Values", fontsize=14)


plt.subplot(122)

importances = rmodel.get_booster().get_score(importance_type='total_gain')
keys = importances.keys()
for k in keys:
    importances[k] = [importances[k]]
importances = pd.DataFrame.from_dict(importances)
importances = importances.T
importances = importances.sort_values(by = 0, ascending=False)
indices =importances.index.values[:10]
importances = importances[0].values[:10]
features = indices
plt.title('(b) Feature Importances \n Total Gain')
plt.barh(range(len(indices)), importances, color='g', align='center')
plt.yticks(range(len(indices)), [i for i in indices])
plt.xlabel('Relative Importance')
plt.gca().invert_yaxis()
#shap.summary_plot(shap_values, X100, plot_type="bar", max_display=10)
plt.show()

plt.tight_layout()





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
        # pdf.savefig()
        # plt.close()
    except:
        print("Issues with SHAP values for {}".format(feat))



data = 1