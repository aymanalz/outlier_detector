import os
import numpy as np
import pandas as pd
import outlier_detector

import matplotlib.pyplot as plt

def add_normal_noise_to_col(df, col, mu=0, seg=1):
    N = len(df)
    noise = np.random.normal(mu, seg, N)
    df[col] = df[col] + noise
    return df


def add_outlier_samples(df, skip_cols,  frac=0.1, random_state = None):
    """
    We assume that df set has x1,x2,..., y
    :param df:
    :return:
    """
    Nnoise = int(frac * len(df))
    df_noise = pd.DataFrame(columns=df.columns)
    for col in df_noise:
        if col in skip_cols:
            continue
        min_val = df[col].min()
        max_val = df[col].max()
        noise = np.random.rand(Nnoise, 1)
        df_noise[col] = min_val + noise.flatten() * (max_val - min_val)

    df = pd.concat([df, df_noise], axis=0)
    df = df.reset_index().drop(["index"], axis=1)
    return df

# ====================================
# Does noise affect model training
# ====================================
if 0:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
    df1 = pd.DataFrame(np.arange(-100, 100, 0.4), columns=['x1'])
    df1['y'] = (-3 * df1['x1'] + np.power(df1['x1'], 2) + np.power(df1['x1'] / 3, 3.0))
    df = df1

    # addnoise on data
    add_normal_noise_to_col(df, 'y', mu=0, seg=1500)
    df['signal'] = 1
    # add noise
    df = add_outlier_samples(df, skip_cols = ['signal'], frac=0.1)
    gb = denoise.xgb_estimator()
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=123)
    gb.fit(df_train[['x1']], df_train['y'])

    ddf_test = df_test[df_test['signal'] == 1]
    y_hat = gb.predict(ddf_test[['x1']])
    plt.scatter(y_hat, ddf_test['y'])
    xx = 1


# ====================================
# Tuning
# ====================================
if 0:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
    from sklearn.model_selection import GridSearchCV

    df1 = pd.DataFrame(np.arange(-100, 100, 0.4), columns=['x1'])
    df1['y'] = (-3 * df1['x1'] + np.power(df1['x1'], 2) + np.power(df1['x1'] / 3, 3.0))
    df = df1

    # addnoise on data
    add_normal_noise_to_col(df, 'y', mu=0, seg=1500)
    df['signal'] = 1
    # add noise
    df = add_outlier_samples(df, skip_cols = ['signal'], frac=0.1)
    gb = denoise.xgb_estimator()
    gb.set_params(seed = 20)
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=123)
    gb.fit(df_train[['x1']], df_train['y'])

    ddf_test = df_test#[df_test['signal'] == 1]
    y_hat = gb.predict(ddf_test[['x1']])
    r2_ = r2_score(y_hat, ddf_test['y'])
    plt.figure()
    plt.scatter(y_hat, ddf_test['y'])
    plt.title("R2 = {}".format(r2_))

    # Tune
    params = {'max_depth': [3, 6, 10],
              'learning_rate': [0.01, 0.05, 0.1],
              'n_estimators': [100, 500, 1000],
              'colsample_bytree': [0.3, 0.7]}

    clf = GridSearchCV(estimator=gb,
                       param_grid=params,
                       scoring='neg_mean_squared_error',
                       verbose=1)
    clf.fit(df_train[['x1']], df_train['y'])
    print("Best parameters:", clf.best_params_)
    print("Lowest RMSE: ", (-clf.best_score_) ** (1 / 2.0))

    gb = denoise.xgb_estimator(clf.best_params_)
    gb.set_params(seed=20)
    gb.fit(df_train[['x1']], df_train['y'])

    plt.figure()
    ddf_test = df_test#[df_test['signal'] == 1]
    y_hat = gb.predict(ddf_test[['x1']])
    r2 = r2_score(y_hat, ddf_test['y'])
    plt.scatter(y_hat, ddf_test['y'])
    plt.title("R2 = {}".format(r2))

    xx = 1

# ====================================
# 1D test problem
# ====================================
if 1:
    df1 = pd.DataFrame(np.arange(-100, 100, 0.4), columns=['x1'])
    df1['y'] = (-3 * df1['x1'] + np.power(df1['x1'], 2) + np.power(df1['x1'] / 3, 3.0))
    df1['y'] = np.random.rand(len(df1))
    df = df1

    # addnoise on data
    add_normal_noise_to_col(df, 'y', mu=0, seg=1500)
    df['signal'] = 1
    # add noise
    df = add_outlier_samples(df, skip_cols = ['signal'], frac=0.7)
    df['id'] = df.index.values
    min_mse =  5.0* (12700004)
    denoise.purify(df, target = 'y', features = ['x1'], col_id = ['id'], max_iterations = 400, min_mse =min_mse)
    stop = 1

if 0:
    from sklearn.datasets import make_moons, make_blobs

    n_samples = 500
    df = 4.0 *( make_moons(n_samples=n_samples, noise=0.05, random_state=0)[0] - np.array([0.5, 0.25]))
    df = pd.DataFrame(df, columns = ['x1', 'y'])
    df['signal'] = 1
    df = add_outlier_samples(df, skip_cols=['signal'], frac=0.1)
    df['id'] = df.index.values
    denoise.purify(df, target='y', features=['x1'], col_id=['id'], max_iterations=400)

# ====================================
# High D test problem
# ====================================
if 0:
    np.random.seed(123)
    samples = 5000
    nfeatures = 10
    features = ['x{}'.format(i) for i in range*nfeatures]
    exponents = [5,2,3,1,7,4,0,-1,6,2]
    coeff = [0.5,-2,3,4,-2.5,0,7,2,3.5]
    df1 = pd.DataFrame(columns=[features])
    for feat in features:
        df1[feat] = 10 * np.random.rand(samples)

    df1['y'] = (-3 * df1['x1'] + np.power(df1['x1'], 2) + np.power(df1['x1'] / 3, 3.0))
    df = df1

    # addnoise on data
    add_normal_noise_to_col(df, 'y', mu=0, seg=1500)

    # add noise
    df = add_outlier_samples(df, frac=0.5)
    df['id'] = df.index.values

    purify(df, target = 'y', features = ['x1'], col_id = ['id'], max_iterations = 400)
    stop = 1
