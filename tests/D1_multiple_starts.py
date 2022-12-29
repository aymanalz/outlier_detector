import os
import pandas as pd
import numpy as np
import outlier_detector
from utils import *
import pickle
import shutil
from joblib import Parallel, delayed

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
                                   symmetry_factor=0.5)
    od.purify(seed=seed)

    fn = os.path.join(result_ws, "run_{}.dat".format(seed))
    fidw = open(fn, 'wb')
    pickle.dump(od, fidw)
    fidw.close()
    return 1

seeds = [576, 3215, 566]
results = Parallel(n_jobs=3)(delayed(single_run)(r) for r in seeds)