import numpy as np
import pandas as pd

def add_normal_noise_to_col(df, col, mu=0, seg=1, random_state = None):
    N = len(df)
    if random_state is None:
        random_state = np.random.RandomState

    noise = random_state.normal(mu, seg, N)

    df[col] = df[col] + noise
    return df


def add_outlier_samples(df, skip_cols,  frac=0.1, random_state = None):
    """
    We assume that df set has x1,x2,..., y
    :param df:
    :return:
    """
    if random_state is None:
        random_state = np.random.RandomState
    Nnoise = int(frac * len(df))
    df_noise = pd.DataFrame(columns=df.columns)
    for col in df_noise:
        if col in skip_cols:
            continue
        min_val = df[col].min()
        max_val = df[col].max()
        noise = random_state.rand(Nnoise, 1)
        df_noise[col] = min_val + noise.flatten() * (max_val - min_val)

    df = pd.concat([df, df_noise], axis=0)
    df = df.reset_index().drop(["index"], axis=1)
    return df