import numpy as np
import pandas as pd
import outlier_detector
from utils import *
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
from joblib import Parallel, delayed


def hart6d(xx):
    # xx = [x1, x2, x3, x4, x5, x6]

    """
     (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573), f(x*) = - 3.32237.
    """
    alpha = [1.0, 1.2, 3.0, 3.2]

    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])

    pm = 1.0*np.array([[1312, 1696, 5569, 124, 8283, 5886],
                   [2329, 4135, 8307, 3736, 1004, 9991],
                   [2348, 1451, 3522, 2883, 3047, 6650],
                   [4047, 8828, 8732, 5743, 1091, 381]])
    P =  pm/10000.0
    outer = 0;
    for ii in range(4):
        inner = 0
        for jj in range(6):
            xj = xx[jj]
            Aij = A[ii, jj]
            Pij = P[ii, jj]
            inner = inner + Aij * np.power((xj - Pij), 2.0)
        new = alpha[ii] * np.exp(-inner)
        outer = outer + new



    return outer

def check_if_xgboost_can_model_this_function(df, features, target):
    random_state = np.random.RandomState(777)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    train_df = df.sample(frac=(0.7), random_state=random_state)
    test_df = df.drop(index=train_df.index)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    params = {
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

    model = xgb.XGBRegressor(**params)

    # fit a model to flag possible samples to removing

    model.set_params(random_state=random_state)
    model.set_params(seed=1111)
    model.fit(train_df[features], train_df[target])

    from sklearn.metrics import r2_score
    y_hat = model.predict(test_df[features])
    plt.scatter(test_df[target], y_hat)
    plt.title(r2_score(test_df[target], y_hat))

    vv = 1

if __name__ == "__main__":

    ratios = [0.25, 0.5, 1.0]


    features = ["X_{}".format(i) for i in range(6)]
    target = 'y'
    random_state = np.random.RandomState(777)
    X = np.random.rand(100000, 6)
    y = np.apply_along_axis(hart6d, 1, X)


    df = pd.DataFrame(X, columns=features)
    df['y'] = y

    if 0:
        add_normal_noise_to_col(df, 'y', mu=0, seg=0.01, random_state=random_state)
        df['signal'] = 1
        df = add_outlier_samples(df, skip_cols=['signal'], frac=0.25, random_state=random_state)
    check_if_xgboost_can_model_this_function(df, features, target)









    cc = 1



