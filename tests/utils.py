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

def split_dataframe_by_position(df, splits):
    """
    Takes a dataframe and an integer of the number of splits to create.
    Returns a list of dataframes.
    """
    dataframes = []
    index_to_split = len(df) // splits
    start = 0
    end = index_to_split
    for split in range(splits):
        temporary_df = df.iloc[start:end, :]
        dataframes.append(temporary_df)
        start += index_to_split
        end += index_to_split

    return dataframes


def add_balanced_outlier_samples(df, skip_cols,  frac=0.1, random_state = None, target = None, splits = 10):
    """
    We assume that df set has x1,x2,..., y
    :param df:
    :return:
    """
    if random_state is None:
        random_state = np.random.RandomState


    df_ = df.sort_values(by='y')
    dfs = split_dataframe_by_position(df_, splits)
    all_noise = []
    for dfff in dfs:
        Nnoise = int(frac * len(dfff))
        df_noise = pd.DataFrame(columns=df.columns)
        for col in df_noise:
            if col in skip_cols:
                continue
            min_val = dfff[col].min()
            max_val = dfff[col].max()
            noise = random_state.rand(Nnoise, 1)
            df_noise[col] = min_val + noise.flatten() * (max_val - min_val)

        all_noise.append(df_noise.copy())
    all_noise = pd.concat(all_noise, axis=0)
    df = pd.concat([df, all_noise], axis=0)
    df = df.reset_index().drop(["index"], axis=1)
    return df

def get_scores(obj, burn_in = 50):
    burn_in = 50
    df = obj.df_results.copy()
    del (df['score'])
    df = df[df['iter'] > burn_in]
    del (df['iter'])

    df = pd.DataFrame(df.mean(axis = 0))
    df.reset_index(inplace=True)
    df.rename(columns={'index': obj.sample_id, 0:'score_mean'}, inplace=True)

    return df