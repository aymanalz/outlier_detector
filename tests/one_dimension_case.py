import os
import pandas as pd
import numpy as np
import outlier_detector
from utils import *
import pickle
import shutil


multi_starts = False
effect_of_test_size = False
effect_of_move_frac = True
effect_of_noise_ratio = True

# ====================================
# 1D test problem
# ====================================
if 0:
    random_state = np.random.RandomState(153)
    df1 = pd.DataFrame(np.arange(-100, 100, 0.4), columns=['x1'])
    df1['y'] = (-3 * df1['x1'] + np.power(df1['x1'], 2) + np.power(df1['x1'] / 3, 3.0))

    df = df1

    # addnoise on data
    add_normal_noise_to_col(df, 'y', mu=0, seg=1500, random_state = random_state)
    df['signal'] = 1
    # add noise
    df = add_outlier_samples(df, skip_cols = ['signal'], frac=0.3, random_state = random_state)
    df['id'] = df.index.values
    min_mse =  5000**2.0


    #‘neg_mean_absolute_error’
    od = outlier_detector.Detector(df,
                                   target = 'y',
                                   features = ['x1'],
                                   sample_id = 'id',
                                   max_iterations = 1000,
                                   min_mse =min_mse,
                                   test_frac=0.3,
                                   damping_weight=0.8,
                                   signal_error_quantile=0.5,
                                   frac_noisy_samples=0.03,
                                   frac_signal_samples=0.03,
                                   score= "neg_mean_squared_error",
                                   proposal_method="quantile",
                                   leakage_rate = 0.015)
    od.purify(seed = 576)

    fn = open("simple_od.dat", 'wb')
    pickle.dump(od, fn)
    fn.close()


# ====================================
# Run algorithm multiple times with different seed numbers
# ====================================
if multi_starts:
    random_state = np.random.RandomState(153)
    df1 = pd.DataFrame(np.arange(-100, 100, 0.4), columns=['x1'])
    df1['y'] = (-3 * df1['x1'] + np.power(df1['x1'], 2) + np.power(df1['x1'] / 3, 3.0))

    df = df1

    # addnoise on data
    add_normal_noise_to_col(df, 'y', mu=0, seg=1500, random_state=random_state)
    df['signal'] = 1
    # add noise
    df = add_outlier_samples(df, skip_cols=['signal'], frac=0.3, random_state=random_state)
    df['id'] = df.index.values
    min_mse = 5000 ** 2.0

    # ‘neg_mean_absolute_error’
    result_ws = os.path.join(r"results", "multi_start")
    if os.path.isdir(result_ws):
        shutil.rmtree(result_ws)
    os.mkdir(result_ws)


    for seed in [576, 3215, 566, 983, 12475 ]:
        od = outlier_detector.Detector(df,
                                       target='y',
                                       features=['x1'],
                                       sample_id='id',
                                       max_iterations=500,
                                       min_mse=min_mse,
                                       test_frac=0.3,
                                       damping_weight=0.8,
                                       signal_error_quantile=0.5,
                                       frac_noisy_samples=0.03,
                                       frac_signal_samples=0.03,
                                       score="neg_mean_squared_error",
                                       proposal_method="quantile",
                                       leakage_rate=0.015)
        od.purify(seed=seed)

        fn = os.path.join(result_ws, "run_{}.dat".format(seed))
        fidw = open(fn, 'wb')
        pickle.dump(od, fidw)
        fidw.close()

if effect_of_test_size:
    random_state = np.random.RandomState(153)
    df1 = pd.DataFrame(np.arange(-100, 100, 0.4), columns=['x1'])
    df1['y'] = (-3 * df1['x1'] + np.power(df1['x1'], 2) + np.power(df1['x1'] / 3, 3.0))

    df = df1

    # addnoise on data
    add_normal_noise_to_col(df, 'y', mu=0, seg=1500, random_state=random_state)
    df['signal'] = 1
    # add noise
    df = add_outlier_samples(df, skip_cols=['signal'], frac=0.3, random_state=random_state)
    df['id'] = df.index.values
    min_mse = 5000 ** 2.0

    # ‘neg_mean_absolute_error’
    result_ws = os.path.join(r"results", "test_fact")
    if os.path.isdir(result_ws):
        shutil.rmtree(result_ws)
    os.mkdir(result_ws)


    for frac in [0.1,0.2,0.3,0.4,0.5,0.6,0.7]:
        od = outlier_detector.Detector(df,
                                       target='y',
                                       features=['x1'],
                                       sample_id='id',
                                       max_iterations=500,
                                       min_mse=min_mse,
                                       test_frac=frac,
                                       damping_weight=0.8,
                                       signal_error_quantile=0.5,
                                       frac_noisy_samples=0.03,
                                       frac_signal_samples=0.03,
                                       score="neg_mean_squared_error",
                                       proposal_method="quantile",
                                       leakage_rate=0.015)
        od.purify(seed=576)

        fn = os.path.join(result_ws, "run_{}.dat".format(frac))
        fidw = open(fn, 'wb')
        pickle.dump(od, fidw)
        fidw.close()

if effect_of_move_frac:
    random_state = np.random.RandomState(153)
    df1 = pd.DataFrame(np.arange(-100, 100, 0.4), columns=['x1'])
    df1['y'] = (-3 * df1['x1'] + np.power(df1['x1'], 2) + np.power(df1['x1'] / 3, 3.0))

    df = df1

    # addnoise on data
    add_normal_noise_to_col(df, 'y', mu=0, seg=1500, random_state=random_state)
    df['signal'] = 1
    # add noise
    df = add_outlier_samples(df, skip_cols=['signal'], frac=0.3, random_state=random_state)
    df['id'] = df.index.values
    min_mse = 5000 ** 2.0

    # ‘neg_mean_absolute_error’
    result_ws = os.path.join(r"results", "candidate_frac")
    if os.path.isdir(result_ws):
        shutil.rmtree(result_ws)
    os.mkdir(result_ws)


    for frac in [0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3]:
        od = outlier_detector.Detector(df,
                                       target='y',
                                       features=['x1'],
                                       sample_id='id',
                                       max_iterations=500,
                                       min_mse=min_mse,
                                       test_frac=0.3,
                                       damping_weight=0.8,
                                       signal_error_quantile=0.5,
                                       frac_noisy_samples= frac,
                                       frac_signal_samples= frac,
                                       score="neg_mean_squared_error",
                                       proposal_method="quantile",
                                       leakage_rate=0.015)
        od.purify(seed=576)

        fn = os.path.join(result_ws, "run_{}.dat".format(frac))
        fidw = open(fn, 'wb')
        pickle.dump(od, fidw)
        fidw.close()

if effect_of_noise_ratio:
    for frac in [0.3, 0.6, 0.9, 1.1]:
        random_state = np.random.RandomState(153)
        df1 = pd.DataFrame(np.arange(-100, 100, 0.4), columns=['x1'])
        df1['y'] = (-3 * df1['x1'] + np.power(df1['x1'], 2) + np.power(df1['x1'] / 3, 3.0))

        df = df1

        # addnoise on data
        add_normal_noise_to_col(df, 'y', mu=0, seg=1500, random_state=random_state)
        df['signal'] = 1
        # add noise
        df = add_outlier_samples(df, skip_cols=['signal'], frac=frac, random_state=random_state)
        df['id'] = df.index.values
        min_mse = 5000 ** 2.0

        # ‘neg_mean_absolute_error’
        result_ws = os.path.join(r"results", "noise_sig_ratio")
        if os.path.isdir(result_ws):
            shutil.rmtree(result_ws)
        os.mkdir(result_ws)

        od = outlier_detector.Detector(df,
                                       target='y',
                                       features=['x1'],
                                       sample_id='id',
                                       max_iterations=500,
                                       min_mse=min_mse,
                                       test_frac=0.3,
                                       damping_weight=0.8,
                                       signal_error_quantile=0.5,
                                       frac_noisy_samples= 0.03,
                                       frac_signal_samples= 0.03,
                                       score="neg_mean_squared_error",
                                       proposal_method="quantile",
                                       leakage_rate=0.015)
        od.purify(seed=576)

        fn = os.path.join(result_ws, "run_{}.dat".format(frac))
        fidw = open(fn, 'wb')
        pickle.dump(od, fidw)
        fidw.close()