import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import outlier_detector
from utils import *
import xgboost as xgb


def train_1d_model(df, features, target, mode='noisy_data', params=None):
    """

    :param df:
    :param features:
    :param target:
    :param mode:
    :param params:
    :return:
    """
    random_state = np.random.RandomState(777)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    train_df = df.sample(frac=(0.7), random_state=random_state)
    test_df = df.drop(index=train_df.index)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    model = xgb.XGBRegressor(**params)

    # fit a model to flag possible samples to removing

    model.set_params(random_state=random_state)
    model.set_params(seed=1111)
    if mode in ['noisy_data']:
        model.fit(train_df[features], train_df[target])
    elif mode in ['flag']:
        df_tr = train_df.copy()
        df_tr = df_tr[df_tr['score_mean'] >= 0.8]
        model.fit(df_tr[features], df_tr[target])
    elif mode in ['weight']:
        df_tr = train_df.copy()
        w = np.exp(20*df_tr['score_mean'].values)
        model.fit(df_tr[features], df_tr[target], sample_weight=w)

    else:
        raise ValueError("Unknown Mode...")

    from sklearn.metrics import r2_score
    if mode in ['noisy_data']:
        y_hat = model.predict(test_df[features])
        test_df['y_hat'] = y_hat
        r2 = r2_score(test_df[target], y_hat)
        return r2, test_df
    elif mode in ['flag']:
        df_tr = test_df.copy()
        df_tr = df_tr[df_tr['score_mean'] >= 0.5]
        y_hat = model.predict(df_tr[features])
        r2 = r2_score(df_tr[target], y_hat)
        df_tr['y_hat'] = y_hat
        test_df['y_hat'] = df_tr['y_hat']
        return r2, test_df
    else:
        df_tr = test_df.copy()
        y_hat = model.predict(df_tr[features])
        w = np.exp(20*df_tr['score_mean'].values)
        r2 = r2_score(df_tr[target], y_hat, sample_weight=w)
        df_tr['y_hat'] = y_hat
        test_df['y_hat'] = df_tr['y_hat']
        return r2, test_df


# ================================
## 1D
# ===============================


# ******* Plot noisy model
random_state = np.random.RandomState(153)
df1 = pd.DataFrame(np.arange(-100, 100, 0.4), columns=['x1'])
df1['y'] = (-3 * df1['x1'] + np.power(df1['x1'], 2) + np.power(df1['x1'] / 3, 3.0))

df = df1

# addnoise on data
add_normal_noise_to_col(df, 'y', mu=0, seg=1500, random_state=random_state)
df['signal'] = 1
# add noise
df = add_outlier_samples(df, skip_cols=['signal'], frac=0.25, random_state=random_state)
df['id'] = df.index.values

features = ['x1']
target = 'y'
params_1d = {

    "objective": "reg:squarederror",
    "tree_method": "hist",
    "colsample_bytree": 0.8,
    "learning_rate": 0.20,
    "max_depth": 7,
    "alpha": 100,
    "n_estimators": 500,
    "subsample": 0.8,
    "reg_lambda": 10,
    "min_child_weight": 5,
    "gamma": 10,
    "max_delta_step": 0,
    "seed": 123,

}
r2, test_df = train_1d_model(df, features, target, params=params_1d)
y_true = test_df[target].copy()
issignal = test_df['signal'].copy()
y_hat = test_df['y_hat'].copy()

plt.subplot(231)
mask_outlier = issignal.isna()
issignal[mask_outlier] = 0

plt.scatter(y_true[mask_outlier], y_hat[mask_outlier], c='r', label='Outliers', s=6)
plt.scatter(y_true[~mask_outlier], y_hat[~mask_outlier], c='b', label='Signal', s=6)
minX = min(y_true.min(), y_hat.min())
maxX = max(y_true.max(), y_hat.max())
plt.plot([minX, maxX], [minX, maxX], 'g', label='1-1 Line')
plt.legend(loc='upper left')
plt.xlabel("True Target Value")
plt.ylabel("Estimated Target Value")
plt.title("(a) 1D Case\n $R^2 = {}$".format(int(1000 * r2) / 1000))
plt.gca().set_aspect('equal', adjustable='box')

# ********************* plot binary model ********************
figures_folder = r"results\figures"
data_folder = r"results\noise_sig_ratio"
fn = os.path.join(data_folder, "noise_ratio_25.dat")

fidr = open(fn, 'rb')
obj = pickle.load(fidr)
fidr.close()

df = obj.df.copy()
df = df.merge(obj.mean_score, how='left', on='id')
r2, test_df = train_1d_model(df, features, target, mode='flag', params=params_1d)

y_true = test_df[target].copy()
issignal = test_df['y_hat'].copy()
y_hat = test_df['y_hat'].copy()

plt.subplot(232)
mask_outlier = issignal.isna()
issignal[mask_outlier] = 0
y_hat[mask_outlier]

minX = min(y_true.min(), y_hat.min())
maxX = max(y_true.max(), y_hat.max())

#plt.scatter(y_true[mask_outlier], y_hat[mask_outlier], c = 'r', label = 'Outliers', s = 6)
plt.scatter(y_true[~mask_outlier], y_hat[~mask_outlier], c='b', label='Signal', s=6)


mask_missed_signal = (test_df['signal']==1) & (test_df['y_hat'].isna())
y_hat[mask_outlier] = minX
plt.scatter(y_true[mask_missed_signal], y_hat[mask_missed_signal], c = 'r', alpha = 1.0,
            label = 'Mislabeled Signal', s = 6, marker = '*')


mask_missed_signal = (test_df['signal']!=1) & (~test_df['y_hat'].isna())
y_hat[mask_outlier] = minX
plt.scatter(y_true[mask_missed_signal], y_hat[mask_missed_signal], c = 'm', alpha = 1.0,
            label = 'Mislabeled Outlier', s = 6,  marker = 's')



plt.plot([minX, maxX], [minX, maxX], 'g', label='1-1 Line')
plt.legend(loc='upper left')
plt.xlabel("True Target Value")
plt.ylabel("Estimated Target Value")
plt.title("(b) 1D Case\n$R^2 = {}$".format(int(1000 * r2) / 1000))
plt.gca().set_aspect('equal', adjustable='box')

# ******************* weights

df = obj.df.copy()
df = df.merge(obj.mean_score, how='left', on='id')
r2, test_df = train_1d_model(df, features, target, mode='weight', params=params_1d)

y_true = test_df[target].copy()
issignal = test_df['y_hat'].copy()
y_hat = test_df['y_hat'].copy()

plt.subplot(233)
mask_outlier = issignal.isna()
issignal[mask_outlier] = 0

# plt.scatter(y_true[mask_outlier], y_hat[mask_outlier], c = 'r', label = 'Outliers', s = 6)
plt.scatter(y_true[~mask_outlier], y_hat[~mask_outlier], c=test_df[~mask_outlier]['score_mean'],
            cmap='jet_r', s=6)
minX = min(y_true.min(), y_hat.min())
maxX = max(y_true.max(), y_hat.max())
plt.plot([minX, maxX], [minX, maxX], 'g', label='1-1 Line')
plt.legend(loc='upper left')
plt.xlabel("True Target Value")
plt.ylabel("Estimated Target Value")
plt.title("(c) 1D Case\n$R^2 = {}$".format(int(1000 * r2) / 1000))
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar()

#### ==============================
# 6D examples
#### =============================
figures_folder = r"results\figures"
data_folder = r"results\noise_sig_ratio"
fn = os.path.join(data_folder, r"D6_noise_ratio_0_25.dat")

fidr = open(fn, 'rb')
obj = pickle.load(fidr)
fidr.close()
df = obj.df.copy()
features = obj.features
target = obj.target

params_6d = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "colsample_bytree": 0.8,
    "learning_rate": 0.060,
    "max_depth": 7,
    "alpha": 0.1,
    "n_estimators": 500,
    "subsample": 0.8,
    "reg_lambda": 1,
    "min_child_weight": 10,
    "gamma": 0.0001,
    "max_delta_step": 0,
    "seed": 123,
}
# train noisy model

r2, test_df = train_1d_model(df, features, target, params=params_6d)
y_true = test_df[target].copy()
issignal = test_df['signal'].copy()
y_hat = test_df['y_hat'].copy()

plt.subplot(234)
mask_outlier = issignal.isna()
issignal[mask_outlier] = 0

plt.scatter(y_true[mask_outlier], y_hat[mask_outlier], c='r', label='Outliers', s=6)
plt.scatter(y_true[~mask_outlier], y_hat[~mask_outlier], c='b', label='Signal', s=6)
minX = min(y_true.min(), y_hat.min())
maxX = max(y_true.max(), y_hat.max())
plt.plot([minX, maxX], [minX, maxX], 'g', label='1-1 Line')
plt.legend(loc='upper left')
plt.xlabel("True Target Value")
plt.ylabel("Estimated Target Value")
plt.title("(d) 6D Case\n $R^2 = {}$".format(int(1000 * r2) / 1000))
plt.gca().set_aspect('equal', adjustable='box')

# binary
df = obj.df.copy()
df = df.merge(obj.mean_score, how='left', on=obj.sample_id)
r2, test_df = train_1d_model(df, features, target, mode='flag', params=params_6d)

y_true = test_df[target].copy()
issignal = test_df['y_hat'].copy()
y_hat = test_df['y_hat'].copy()

plt.subplot(235)
mask_outlier = issignal.isna()
issignal[mask_outlier] = 0

minX = min(y_true.min(), y_hat.min())
maxX = max(y_true.max(), y_hat.max())

#plt.scatter(y_true[mask_outlier], y_hat[mask_outlier], c = 'r', label = 'Outliers', s = 6)
plt.scatter(y_true[~mask_outlier], y_hat[~mask_outlier], c='b', label='Signal',
            s=6)


mask_missed_signal = (test_df['signal']==1) & (test_df['y_hat'].isna())
y_hat[mask_outlier] = minX
plt.scatter(y_true[mask_missed_signal], y_hat[mask_missed_signal], c = 'r', alpha = 1.0,
            label = 'Mislabeled Signal', s = 6, marker = '*')


mask_missed_signal = (test_df['signal']!=1) & (~test_df['y_hat'].isna())
y_hat[mask_outlier] = minX
plt.scatter(y_true[mask_missed_signal], y_hat[mask_missed_signal], c = 'm', alpha = 1.0,
            label = 'Mislabeled Outlier', s = 6,  marker = 's')


plt.plot([minX, maxX], [minX, maxX], 'g', label='1-1 Line')
plt.legend(loc='upper left')
plt.xlabel("True Target Value")
plt.ylabel("Estimated Target Value")
plt.title("(e) 6D Case\n$R^2 = {}$".format(int(1000 * r2) / 1000))
plt.gca().set_aspect('equal', adjustable='box')

# ******************* weights

df = obj.df.copy()
df = df.merge(obj.mean_score, how='left', on=obj.sample_id)
r2, test_df = train_1d_model(df, features, target, mode='weight', params=params_6d)

y_true = test_df[target].copy()
issignal = test_df['y_hat'].copy()
y_hat = test_df['y_hat'].copy()

plt.subplot(236)
mask_outlier = issignal.isna()
issignal[mask_outlier] = 0

# plt.scatter(y_true[mask_outlier], y_hat[mask_outlier], c = 'r', label = 'Outliers', s = 6)
plt.scatter(y_true[~mask_outlier], y_hat[~mask_outlier], c=test_df[~mask_outlier]['score_mean'],
            cmap='jet_r', s=6)
minX = min(y_true.min(), y_hat.min())
maxX = max(y_true.max(), y_hat.max())
plt.plot([minX, maxX], [minX, maxX], 'g', label='1-1 Line')
plt.legend(loc='upper left')
plt.xlabel("True Target Value")
plt.ylabel("Estimated Target Value")
plt.title("(f) 6D Case\n$R^2 = {}$".format(int(1000 * r2) / 1000))
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')

cc = 1
