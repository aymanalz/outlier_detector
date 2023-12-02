# ===============****===================
"""
Annual Water Use Model
Date:
"""
# ===============****===================
from lightgbm import LGBMRegressor
import os
import warnings
import pandas as pd
import numpy as np
from iwateruse import (
    data_cleaning,
    report,
    splittors,
    pre_train_utils,
    make_dataset,
    figures,
)
from iwateruse import denoise, model_diagnose
import matplotlib.pyplot as plt
import xgboost as xgb
import json
import shap
import joblib
from sklearn.pipeline import Pipeline
from iwateruse.model import Model
from iwateruse import (
    targets,
    weights,
    pipelines,
    outliers_utils,
    estimators,
    featurize,
    predictions,
)
from iwateruse import selection, interpret
from sklearn.model_selection import train_test_split
import copy
import pickle
import outlier_detector
from joblib import Parallel, delayed

# Seeds
split_seed = 123



# ========================================
# Loading data
# ========================================
df_train = pd.read_csv(r"south_westh.csv")
del(df_train['Unnamed: 0'])
target = 'per_capita'
features = []
fidr = open(r"reduced_features.txt", 'r')
feat = fidr.readlines()
fidr.close()

for ft in feat:
    f = ft.strip()
    f = f.replace("'", "")
    f = f.replace(",", "")
    features.append(f)

df_train = df_train[features + [target]]
X_train, X_test, y_train, y_test = train_test_split(
    df_train,
    df_train[target],
    test_size=0.3,
    shuffle=True,
    random_state=split_seed
)

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

gb = xgb.XGBRegressor(**params)

curr_trained_model = gb.fit(X_train[features], y_train)
y_pred = curr_trained_model.predict(X_test[features])




df_train['sample_id'] = list(range(len(df_train)))
df_train = df_train.sample(frac = 0.9)

def detect(seed):
    min_mse = 30**2.0
    od = outlier_detector.Detector(df_train,
                                   target =target,
                                   features = features,
                                   sample_id = 'sample_id',
                                   max_iterations = 1000,
                                   min_mse =min_mse,
                                   test_frac=0.3,
                                   damping_weight=0.8,
                                   signal_error_quantile=0.5,
                                   frac_noisy_samples=0.03,
                                   frac_signal_samples=0.03,
                                   score= "neg_mean_squared_error",
                                   proposal_method="quantile",
                                   leakage_rate = 0.01,
                                   symmetry_factor=0.33,
                                   ml_hyperparamters = params)
    od.purify(seed = seed)
    fn = open("ca_wu_od_{}.dat".format(seed), 'wb')
    pickle.dump(od, fn)
    fn.close()
    return 1
seeds = [1253, 5523, 8891, 2130, 52347]
results = Parallel(n_jobs=5)(delayed(detect)(r) for r in seeds)


