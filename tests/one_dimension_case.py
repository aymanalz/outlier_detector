import os
import pandas as pd
import numpy as np
import outlier_detector
from utils import *
import pickle
import shutil
from joblib import Parallel, delayed


Simple_test = False
multi_starts = True
effect_of_noise_ratio = True
effect_of_test_size = True
effect_of_update_step = True
effect_of_proposal_function = True


# ====================================
# 1D test problem
# ====================================


if Simple_test:
    random_state = np.random.RandomState(153)
    df1 = pd.DataFrame(np.arange(-100, 100, 0.4), columns=['x1'])
    df1['y'] = (-3 * df1['x1'] + np.power(df1['x1'], 2) + np.power(df1['x1'] / 3, 3.0))

    df = df1

    # addnoise on data
    add_normal_noise_to_col(df, 'y', mu=0, seg=1500, random_state = random_state)
    df['signal'] = 1
    # add noise
    df = add_outlier_samples(df, skip_cols = ['signal'], frac=0.25, random_state = random_state)
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
                                   frac_noisy_samples=0.05,
                                   frac_signal_samples=0.05,
                                   score= "neg_mean_squared_error",
                                   proposal_method="quantile",
                                   leakage_rate = 0.02,
                                   symmetry_factor=0.3)
    od.purify(seed = 576)

    fn = open("simple_od.dat", 'wb')
    pickle.dump(od, fn)
    fn.close()

if multi_starts:
    def single_run(seed):
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
        od = outlier_detector.Detector(df,
                                       target='y',
                                       features=['x1'],
                                       sample_id='id',
                                       max_iterations=1000,
                                       min_mse=min_mse,
                                       test_frac=0.3,
                                       damping_weight=0.8,
                                       signal_error_quantile=0.5,
                                       frac_noisy_samples=0.03,
                                       frac_signal_samples=0.03,
                                       score="neg_mean_squared_error",
                                       proposal_method="quantile",
                                       leakage_rate=0.015,
                                       symmetry_factor=0.3)
        od.purify(seed=seed)

        fn = os.path.join(result_ws, "run_{}.dat".format(seed))
        fidw = open(fn, 'wb')
        pickle.dump(od, fidw)
        fidw.close()
        return 1

    seeds = [576, 3215, 566]
    results = Parallel(n_jobs=3)(delayed(single_run)(r) for r in seeds)

if effect_of_noise_ratio:
    def detect_n2s_ratio(n2s_ratio):
        random_state = np.random.RandomState(153)
        df1 = pd.DataFrame(np.arange(-100, 100, 0.4), columns=['x1'])
        df1['y'] = (-3 * df1['x1'] + np.power(df1['x1'], 2) + np.power(df1['x1'] / 3, 3.0))

        df = df1

        # addnoise on data
        add_normal_noise_to_col(df, 'y', mu=0, seg=1500, random_state = random_state)
        df['signal'] = 1
        # add noise
        df = add_outlier_samples(df, skip_cols = ['signal'], frac=n2s_ratio, random_state = random_state)
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
                                       leakage_rate = 0.015,
                                       symmetry_factor=0.3)
        od.purify(seed = 576)

        fn = open("results/noise_sig_ratio/noise_ratio_{}.dat".format(int(n2s_ratio*100)), 'wb')
        pickle.dump(od, fn)
        fn.close()
        return 1


    noise_2_signal_ratio = [0.25, 0.5, 1.0]
    results = Parallel(n_jobs=3)(delayed(detect_n2s_ratio)(r) for r in noise_2_signal_ratio)

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


    def detect_test_size(frac):
        od = outlier_detector.Detector(df,
                                       target='y',
                                       features=['x1'],
                                       sample_id='id',
                                       max_iterations=1000,
                                       min_mse=min_mse,
                                       test_frac=frac,
                                       damping_weight=0.8,
                                       signal_error_quantile=0.5,
                                       frac_noisy_samples=0.03,
                                       frac_signal_samples=0.03,
                                       score="neg_mean_squared_error",
                                       proposal_method="quantile",
                                       leakage_rate=0.015,
                                       symmetry_factor=0.3)
        od.purify(seed=576)
        fn = os.path.join(result_ws, "run_{}.dat".format(frac))
        fidw = open(fn, 'wb')
        pickle.dump(od, fidw)
        fidw.close()
        return 1
    fracs =  [0.15, 0.3, 0.45]
    results = Parallel(n_jobs=3)(delayed(detect_test_size)(r) for r in fracs)

if effect_of_update_step:
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
    min_mse = 5000 ** 2.0

    # ‘neg_mean_absolute_error’
    result_ws = os.path.join(r"results", "update_step")
    if os.path.isdir(result_ws):
        shutil.rmtree(result_ws)
    os.mkdir(result_ws)

    def detect_step_size(frac):
        od = outlier_detector.Detector(df,
                                       target='y',
                                       features=['x1'],
                                       sample_id='id',
                                       max_iterations=1000,
                                       min_mse=min_mse,
                                       test_frac=0.3,
                                       damping_weight=0.8,
                                       signal_error_quantile=0.5,
                                       frac_noisy_samples= frac,
                                       frac_signal_samples= frac,
                                       score="neg_mean_squared_error",
                                       proposal_method="quantile",
                                       leakage_rate=0.015,
                                       symmetry_factor=0.3)
        od.purify(seed=576)
        fn = os.path.join(result_ws, "run_{}.dat".format(frac))
        fidw = open(fn, 'wb')
        pickle.dump(od, fidw)
        fidw.close()
        return 1


    fracs = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3]
    results = Parallel(n_jobs=6)(delayed(detect_step_size)(r) for r in fracs)

if effect_of_proposal_function:
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
    min_mse = 5000 ** 2.0

    # ‘neg_mean_absolute_error’
    def detec_sampler(sampler):
        od = outlier_detector.Detector(df,
                                       target='y',
                                       features=['x1'],
                                       sample_id='id',
                                       max_iterations=1000,
                                       min_mse=min_mse,
                                       test_frac=0.3,
                                       damping_weight=0.8,
                                       signal_error_quantile=0.5,
                                       frac_noisy_samples=0.03,
                                       frac_signal_samples=0.03,
                                       score="neg_mean_squared_error",
                                       proposal_method=sampler,
                                       leakage_rate=0.015,
                                       symmetry_factor=0.3)
        od.purify(seed=576)

        fn = open("results/smaplers/{}.dat".format(sampler), 'wb')
        pickle.dump(od, fn)
        fn.close()
        return 1


    funcp = ['quantile', 'mse', 'random_walk']
    results = Parallel(n_jobs=3)(delayed(detec_sampler)(r) for r in funcp)


