import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from numpy.random import choice
import time
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


def _get_args(arg, default=None, **kwargs):
    if arg in list(kwargs.keys()):
        val = kwargs[arg]
        if val is None:
            val = default
    else:
        val = default
    return val


def xgb_estimator(params={}):
    if params is None:
        params = {
            'objective': "reg:squarederror",
            'tree_method': 'hist',
            'colsample_bytree': 0.8,
            'learning_rate': 0.20,
            'max_depth': 7,
            'alpha': 100,
            'n_estimators': 500,
            'rate_drop': 0.9,
            'skip_drop': 0.5,
            'subsample': 0.8,
            'reg_lambda': 10,
            'min_child_weight': 5,
            'gamma': 10,
            'max_delta_step': 0,
            'seed': 123
        }

    gb = xgb.XGBRegressor(**params)
    return gb


def compute_sampling_weight(err, error_quantile, outliers_sampling_method='equal_weight'):
    if outliers_sampling_method in ['equal_weight']:
        qq = np.quantile(err, error_quantile)
        w = np.zeros_like(err)
        w[err >= qq] = 1
        w = w / np.sum(w)
    elif outliers_sampling_method in ['mse']:
        w = err / np.sum(err)
    else:
        raise ValueError("Unkown method")
    return w, qq


def propose_sample_removal(df_, features, target, test_frac=0.3, outliers_sampling_method='equal_weight',
                           error_quantile=0.70, frac_noisy_samples=0.01, kfolds=5,
                           score='neg_mean_absolute_percentage_error', xgb_params=None, min_rel_error=0.05):
    """

    """
    # shuffle and split
    df = df_.copy()
    df = df.sample(frac=1).reset_index(drop=True)
    train_df = df.sample(frac=(1 - test_frac))
    test_df = df.drop(index=train_df.index)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # fit a model to flag possible samples to removing
    gb1 = xgb_estimator(params=xgb_params)
    gb1.fit(train_df[features], train_df[target])
    err = np.power((gb1.predict(test_df[features]) - test_df[target]), 2.0)

    w, qq = compute_sampling_weight(err, error_quantile, outliers_sampling_method=outliers_sampling_method)
    relative_error = (np.abs(gb1.predict(test_df[features]) - test_df[target])) / np.abs(test_df[target])
    rqq = np.quantile(relative_error, error_quantile)
    w[relative_error < min_rel_error] = 0
    w = w / np.sum(w)
    # propose samples to remove
    nn = int(len(test_df.index.values) * frac_noisy_samples)
    if nn > len(w[w > 0]):
        nn = len(w[w > 0])

    move_samples = choice(test_df.index.values, nn, p=w, replace=False)

    df_to_move = test_df.iloc[move_samples]
    test_df_new = test_df.drop(index=move_samples)
    new_df = pd.concat([train_df, test_df_new], axis=0)
    new_df.reset_index(inplace=True)
    del (new_df['index'])

    gb = xgb_estimator(params=xgb_params)
    signal_scores = cross_val_score(gb, new_df[features], new_df[target],
                                    scoring=score, cv=kfolds)
    new_score = np.mean(signal_scores)
    return new_score, new_df, df_to_move, gb1, qq, rqq


def damping(old_value, new_value, weight):
    score = (weight * old_value + new_value) / (weight + 1)
    return score


def visulize(fig=None, axs=None,
             signal_average_scroe=None, sigal_smoothed_score=None,
             iter=0, frac_noise_list=None
             ):
    if fig is None:
        fig, axs = plt.subplots(ncols=2, nrows=1)
        return fig, axs

    plt.ion()

    plot1, = axs[0].plot(np.log10(np.abs(signal_average_scroe)), 'r', label='Error in Noise Model')
    plot2, = axs[0].plot(np.log10(np.abs(sigal_smoothed_score)), 'b', label='Error in Signal Model')
    plot3, = axs[1].plot(frac_noise_list, 'b', label='Error in Signal Model')
    if iter == 0:
        axs[0].legend(loc='upper right', bbox_to_anchor=(0, 0.5))

    plot1.set_ydata(np.log10(np.abs(signal_average_scroe)))
    plot2.set_ydata(np.log10(np.abs(sigal_smoothed_score)))

    axs[0].set_title(str(iter))
    fig.canvas.draw()
    time.sleep(0.1)
    fig.canvas.flush_events()

    plt.tight_layout()
    plt.show()


def purify(df, **kwargs):
    initial_split_frac = _get_args('initial_split_frac', default=0.5, **kwargs)
    target = _get_args('target', default='y', **kwargs)
    features = _get_args('features', default=df.columns, **kwargs)
    if target in features:
        features.remove(target)

    col_id = _get_args('col_id', default=df.columns, **kwargs)
    score = _get_args('score', default='neg_mean_absolute_percentage_error', **kwargs)
    N = _get_args('max_iterations', default=200, **kwargs)
    kfolds = _get_args('kfolds', default=5, **kwargs)
    test_frac = _get_args('test_frac', default=0.3, **kwargs)
    max_signal_ratio = _get_args('max_signal_ratio', default=30, **kwargs)
    min_signal_ratio = _get_args('min_signal_ratio', default=0.5, **kwargs)
    cooling_rate = _get_args('cooling_rate', default=0.999, **kwargs)
    damping_weight = _get_args('damping_weight', default=5.0, **kwargs)
    estimator = _get_args('estimator', default=xgb_estimator(params=None), **kwargs)
    min_mse = _get_args('min_mse', default=xgb_estimator(params=None), **kwargs)

    kwargs['initial_df'] = df.copy()
    fig, axs = visulize(fig=None)

    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    df_signal = df.sample(frac=initial_split_frac)
    df_noise = df.drop(index=df_signal.index)

    signal_scores = cross_val_score(estimator, df_signal[features], df_signal[target],
                                    scoring=score, cv=kfolds)
    signal_average_score = [np.mean(signal_scores)]

    frac_noise_list = []
    sigal_smoothed_score = []
    signal_iter_score = []
    results = []
    window = 1
    min_score = 0
    frac_noisy_samples = 0.05
    frac_signal_samples = 0.05
    noise_gammas = []
    signal_gammas = []
    error_quantile = 0.90
    min_rel_error = 0.05
    for iter in range(N):
        # remove outliers from  signal

        new_score, new_df, df_move, signal_model, error_upper_threshold, err_up_rel = \
            propose_sample_removal(df_signal,
                                 features,
                                 target,
                                 test_frac=test_frac,
                                 frac_noisy_samples=frac_noisy_samples,
                                 xgb_params=estimator.get_params(),
                                 score=score,
                                 error_quantile=error_quantile,
                                 min_rel_error=min_rel_error)
        if error_upper_threshold < min_mse:
            error_upper_threshold = min_mse
        gamma = signal_average_score[-1] / new_score
        signal_gammas.append(new_score)

        window = window * cooling_rate

        if window < 0.1:
            window = 0.1
        if window > 1.0:
            window = 1.0

        u = np.random.rand(1)
        signal_frac = len(df_signal) / len(df_noise)
        if u <= gamma:
            if signal_frac > min_signal_ratio:
                new_score = damping(signal_average_score[-1], new_score, damping_weight)
                df_noise = pd.concat([df_noise, df_move], axis=0)
                df_noise.reset_index(inplace=True)
                del (df_noise['index'])
                df_signal = new_df
                signal_average_score.append(new_score)
        else:
            signal_average_score.append(signal_average_score[-1])

        # ================================
        # remove signal from outlier
        # ================================
        df_noise['err'] = np.power((signal_model.predict(df_noise[features]) - df_noise[target]), 2)

        ### *****************
        # propose samples to remove
        err = df_noise['err'].values
        rel_err = np.abs((signal_model.predict(df_noise[features]) - df_noise[target])) / np.abs(df_noise[target])
        w = np.zeros_like(rel_err)
        if err_up_rel < min_rel_error:
            err_up_rel = min_rel_error
        if 1:
            w[err < error_upper_threshold] = 1.0
            # w[rel_err<min_rel_error] = 0
            w = w / np.sum(w)
            w[np.isnan(w)] = 0
        else:
            w = 1 / np.power(err, 0.5)
            w = w / np.sum(w)

        nn = int(len(df_noise.index.values) * frac_signal_samples)
        if nn > len(w[w > 0]):
            nn = len(w[w > 0])
        if nn > 0:
            move_samples = choice(df_noise.index.values, nn, p=w, replace=False)
        else:
            move_samples = []

        df_move = df_noise.iloc[move_samples]
        df_noise_new = df_noise.drop(index=move_samples)
        df_noise_new.reset_index(inplace=True)
        del (df_noise_new['index'])

        df_signal_ = pd.concat([df_signal, df_move], axis=0)
        df_signal_.reset_index(inplace=True)
        del (df_signal_['index'])

        signal_scores = cross_val_score(estimator, df_signal_[features], df_signal_[target],
                                        scoring=score, cv=kfolds)
        new_score = np.mean(signal_scores)

        u = np.random.rand(1)
        gamma = signal_average_score[-1] / new_score
        signal_frac = len(df_signal) / len(df_noise)

        if u <= gamma * window:

            if signal_frac < max_signal_ratio:
                df_signal = df_signal_.copy()
                signal_average_score.append(new_score)
                del (df_noise_new['err'])
                df_noise = df_noise_new.copy()

        else:
            del (df_noise['err'])
            signal_average_score.append(signal_average_score[-1])

        frac_noise_list.append(1.0 / signal_frac)
        if len(signal_average_score) > 1:
            ave_score = (signal_average_score[-1] + signal_average_score[-2]) / 2.0
            signal_iter_score.append(ave_score)
        else:
            signal_iter_score.append(signal_average_score[-1])

        if len(signal_iter_score) > 10:
            last5 = np.array(signal_iter_score)[-10:]
            sigal_smoothed_score.append(np.mean(last5))
        else:
            last5 = np.array(signal_iter_score)
            sigal_smoothed_score.append(np.mean(last5))

        if iter == 0:
            columns = ['iter', 'score', 'sscore'] + list(range(len(df)))
            df_results = pd.DataFrame(np.nan, index=list(range(N)), columns=columns)
            df_results['iter'] = np.arange(N)
        iter_mask = df_results['iter'] == iter
        df_results.loc[iter_mask, 'score'] = signal_iter_score[-1]
        df_results.loc[iter_mask, 'sscore'] = sigal_smoothed_score[-1]
        signal_ids = df_signal['id'].values.tolist()
        df_results.loc[iter_mask, list(range(len(df)))] = 0
        df_results.loc[iter_mask, signal_ids] = 1

        if np.mod(iter, 10) == 0:
            sigs = df_signal.sample(frac=0.01)
            noss = df_noise.sample(frac=0.01)

            df_signal = df_signal.drop(sigs.index)
            df_signal = pd.concat([df_signal, noss])
            df_signal.reset_index(inplace=True)
            del (df_signal['index'])

            df_noise = df_noise.drop(noss.index)
            df_noise = pd.concat([df_noise, sigs])
            df_noise.reset_index(inplace=True)
            del (df_noise['index'])

        if 1:
            visulize(
                fig=fig, axs=axs,
                signal_average_scroe=signal_iter_score, sigal_smoothed_score=sigal_smoothed_score,
                iter=iter, frac_noise_list=frac_noise_list
            )

    return df_results
    x = 1
