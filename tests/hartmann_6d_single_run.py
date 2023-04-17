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



    def detect(ratio):
        features = ["X_{}".format(i) for i in range(6)]
        target = 'y'
        random_state = np.random.RandomState(777)
        X = np.random.rand(10000, 6)
        y = np.apply_along_axis(hart6d, 1, X)


        df = pd.DataFrame(X, columns=features)
        df['y'] = np.log10(y)
        max_sigal = df['y'].max()
        min_signa = df['y'].min()

        add_normal_noise_to_col(df, 'y', mu=0, seg=0.1, random_state=random_state)
        df['signal'] = 1
        #df = add_outlier_samples(df, skip_cols=['signal'], frac=ratio, random_state=random_state)
        df = add_balanced_outlier_samples(df, skip_cols=['signal'], frac=ratio, random_state=random_state,
                                          target='y', splits=50)
        df_outliers = df[df['signal'].isna()].copy()
        threshold = 0.8
        if 0:
            cols = df_outliers.columns
            Xo = df_outliers[features].values
            df_outliers['true_y'] = np.log10(np.apply_along_axis(hart6d, 1, Xo))
            df_outliers['a1'] = min_signa
            df_outliers['a2'] = df_outliers['true_y'] - threshold
            df_outliers.loc[df_outliers['a2']<min_signa, 'a2'] = min_signa
            df_outliers.loc[df_outliers['a2'] > max_sigal, 'a2'] = max_sigal
            df_outliers['a3'] = df_outliers['true_y'] + threshold
            df_outliers.loc[df_outliers['a3'] < min_signa, 'a3'] = min_signa
            df_outliers.loc[df_outliers['a3'] > max_sigal, 'a3'] = max_sigal
            df_outliers['a4'] = max_sigal
            df_outliers['p1'] = (df_outliers['a2'] - df_outliers['a1'])/(df_outliers['a2'] - df_outliers['a1']+
                                                                         df_outliers['a4'] - df_outliers['a3'])
            df_outliers['u'] = np.random.rand(len(df_outliers))
            masklower = df_outliers['u']<=df_outliers['p1']
            a2_a1 = (df_outliers['a2']-df_outliers['a1'])
            p1 = df_outliers['p1']
            u1 = df_outliers['u']
            ylower = df_outliers['a1'] + (a2_a1*u1/p1)
            #df_outliers.loc[masklower, 'rand_y'] = ylower[masklower]
            p2 = 1-df_outliers['p1']
            u2 = df_outliers['u'] - df_outliers['p1']
            a4_a3 = df_outliers['a4'] - df_outliers['a3']
            yupper = df_outliers['a3'] + u2 * p2/a4_a3

            df_outliers.loc[masklower, 'rand_y'] = ylower[masklower]
            df_outliers.loc[~masklower, 'rand_y'] = yupper[~masklower]
            df_outliers['y'] = df_outliers['rand_y']

            df_outliers = df_outliers[cols]
            df = pd.concat([df[~df['signal'].isna()],
                            df_outliers])
        else:
            cols = df_outliers.columns
            Xo = df_outliers[features].values
            df_outliers['true_y'] = np.log10(np.apply_along_axis(hart6d, 1, Xo))
            err = df_outliers['y'] - df_outliers['true_y']

            mask = (np.abs(err) < threshold) & (err <=0)
            shift = df_outliers['y'] - threshold
            df_outliers.loc[mask, 'y'] = shift[mask]

            mask = (np.abs(err) < threshold) & (err > 0)
            shift = df_outliers['y'] + threshold
            df_outliers.loc[mask, 'y'] = shift[mask]
            df_outliers = df_outliers[cols]
            df = pd.concat([df[~df['signal'].isna()],
                            df_outliers])



        check_if_xgboost_can_model_this_function(df, features, target)

        min_mse = 0.1**2
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

        # ‘neg_mean_absolute_error’
        od = outlier_detector.Detector(df,
                                       target='y',
                                       features=features,
                                       max_iterations=500,
                                       min_mse=min_mse,
                                       test_frac=0.3,
                                       damping_weight=0.8,
                                       signal_error_quantile=0.5,
                                       frac_noisy_samples=0.05,
                                       frac_signal_samples=0.05,
                                       score="neg_mean_squared_error",
                                       proposal_method="quantile",
                                       leakage_rate=0.02,
                                       symmetry_factor=0.5,
                                       ml_hyperparamters= params
                                       )
        od.purify(seed=576)
        fn = open("Single_D6_noise_ratio_0_{}.dat".format(int(100*ratio)), 'wb')
        pickle.dump(od, fn)
        fn.close()
        return 1


    detect(0.25)
    cc = 1






    cc = 1



