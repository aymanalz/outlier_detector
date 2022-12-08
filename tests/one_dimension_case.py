import pandas as pd
import numpy as np
import outlier_detector
from utils import *


# ====================================
# 1D test problem
# ====================================
if 1:
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
    min_mse =  10000**2.0

    """
    # good results
    max_iterations = 400,
    min_mse =min_mse,
    test_frac=0.3,
    damping_weight=0.833,
    signal_error_quantile=0.90,
    frac_noisy_samples=0.01,
    frac_signal_samples=0.01,
    """
    #‘neg_mean_absolute_error’
    od = outlier_detector.Detector(df,
                                   target = 'y',
                                   features = ['x1'],
                                   max_iterations = 400,
                                   min_mse =min_mse,
                                   test_frac=0.3,
                                   damping_weight=0.0,
                                   signal_error_quantile=0.5,
                                   frac_noisy_samples=0.1,
                                   frac_signal_samples=0.1,
                                   score= "neg_mean_squared_error")
    od.purify(seed = 576)
    stop = 1