import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from numpy.random import choice
import time
import scipy
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from numpy.random import RandomState

class Detector(object):
    def __init__(self, df,
                 target = 'y',
                 features = ['x1'],
                 initial_split_frac = 0.5,
                 max_iterations = 400,
                 min_mse = None,
                 score = 'neg_mean_absolute_percentage_error',
                 kfolds = 5,
                 test_frac = 0.3,
                 max_signal_ratio = 30,
                 min_signal_ratio = 0.5,
                 cooling_rate = 0.999,
                 damping_weight = 0.8333,
                 estimator = None,
                 signal_error_quantile = 0.80,
                 frac_noisy_samples = 0.01,
                 frac_signal_samples = 0.01,
                 ml_hyperparamters = None

                 ):


        self.df = df
        self.initial_split_frac = initial_split_frac
        self.target = target
        self.features = features
        if self.target in self.features:
            self.features.remove(target)

        self.score = score
        self.max_iterations = max_iterations
        self.min_mse = min_mse
        self.kfolds = kfolds
        self.test_frac = test_frac
        self.max_signal_ratio = max_signal_ratio
        self.min_signal_ratio = min_signal_ratio
        self.cooling_rate = cooling_rate
        self.damping_weight = damping_weight
        self.initial_df = self.df.copy()

        self.outliers_sampling_method = 'equal_weight'
        self.signal_error_quantile = signal_error_quantile
        self.frac_noisy_samples = frac_noisy_samples
        self.frac_signal_samples = frac_signal_samples

        if ml_hyperparamters is None:
            self.ml_hyperparamters =  {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "colsample_bytree": 0.8,
            "learning_rate": 0.20,
            "max_depth": 7,
            "alpha": 100,
            "n_estimators": 500,
            "subsample": 0.8,
            "reg_lambda": 10,
            "min_child_weight": 5,
            "gamma": 10,
            "max_delta_step": 0,
            "seed": 123,
        }

        else:
            self.ml_hyperparamters = ml_hyperparamters

        if estimator is None:
            self.estimator = self.xgb_estimator(params=None)



    def _get_args(self, arg, default=None, **kwargs):
        if arg in list(kwargs.keys()):
            val = kwargs[arg]
            if val is None:
                val = default
        else:
            val = default
        return val


    def xgb_estimator(self, params= None):
        if params is None:
            params = self.ml_hyperparamters

        gb = xgb.XGBRegressor(**params)
        return gb


    def compute_sampling_weight(self, err):
        outliers_sampling_method = self.outliers_sampling_method
        if outliers_sampling_method in ['equal_weight']:
            qq = np.quantile(err, self.signal_error_quantile)
            w = np.zeros_like(err)
            w[err >= qq] = 1
            w = w / np.sum(w)
        elif outliers_sampling_method in ['mse']:
            w = err / np.sum(err)
        else:
            raise ValueError("Unkown method")
        return w, qq


    def propose_sample_removal(self, min_rel_error=0.05):
        """

        """
        # shuffle and split
        error_quantile = self.signal_error_quantile
        kfolds = self.kfolds
        df_ = self.df_signal
        score = self.score

        df = df_.copy()
        df = df.sample(frac=1,random_state = self.get_seed()).reset_index(drop=True)
        train_df = df.sample(frac=(1 - self.test_frac), random_state = self.get_seed())
        test_df = df.drop(index=train_df.index)
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        # fit a model to flag possible samples to removing
        self.signal_model = self.xgb_estimator(params=self.ml_hyperparamters)
        self.signal_model.set_params(random_state=self.get_seed())
        self.signal_model.set_params(seed=self.get_seed())
        self.signal_model.fit(train_df[self.features], train_df[self.target])
        err = np.power((self.signal_model.predict(test_df[self.features]) - test_df[self.target]), 2.0)

        w, self.max_signal_error = self.compute_sampling_weight(err)
        relative_error = (np.abs(self.signal_model.predict(test_df[self.features]) - test_df[self.target])) / np.abs(test_df[self.target])
        w[relative_error < min_rel_error] = 0
        w = w / np.sum(w)

        # propose samples to remove
        nn = int(len(test_df.index.values) * self.frac_noisy_samples)
        if nn > len(w[w > 0]):
            nn = len(w[w > 0])

        np.random.seed(self.get_seed())
        move_samples = choice(test_df.index.values, nn, p=w, replace=False)

        df_to_move = test_df.iloc[move_samples]
        test_df_new = test_df.drop(index=move_samples)
        new_df = pd.concat([train_df, test_df_new], axis=0)
        new_df.reset_index(inplace=True)
        del (new_df['index'])

        gb = self.xgb_estimator(params=self.ml_hyperparamters)
        gb.set_params(random_state=self.get_seed())
        gb.set_params(seed=self.get_seed())


        signal_scores = cross_val_score(gb,
                                        new_df[self.features],
                                        new_df[self.target],
                                        scoring=score, cv=kfolds)
        new_score = np.mean(signal_scores)
        self.df_s2o = df_to_move
        self.propose_df_signal = new_df

        return new_score


    def damping(self, old_value, new_value, weight):
        score = (weight * old_value + (1-weight)*new_value)
        return score


    def visulize(self, fig=None, axs=None,
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

    def cross_validate(self,
                       estimator = None,
                       df_feat = None,
                       df_target = None,
                       scoring = None,
                       cv = None):
        if estimator is None:
            estimator = self.estimator
        if df_feat is None:
            df_feat = self.df_signal[self.features]
        if df_target is None:
            df_target = self.df_signal[self.target]
        if scoring is None:
            scoring = self.score

        #cv = sklearn.model_selection.KFold(n_splits=self.kfolds) #,random_state=self.get_seed(), shuffle = True

        estimator.set_params(random_state=self.get_seed())
        estimator.set_params(seed =self.get_seed())
        cv_score = cross_val_score(estimator,
                        df_feat,
                        df_target,
                        scoring=scoring,
                        cv=self.kfolds)


        return cv_score

    def get_seed(self):
        seed = int(1e6 * self.RandomState.rand())
        return seed


    def purify(self, seed = 123):
        """

        :param df:
        :param kwargs:
        :return:
        """

        fig, axs = self.visulize(fig=None)
        self.RandomState = np.random.RandomState(seed)

        # ===============================================================
        # Shuffle, split, and initial cross validate
        # ===============================================================
        self.df = self.df.sample(frac=1, random_state =  self.get_seed() ).reset_index(drop=True)
        self.df_signal = self.df.sample(frac=self.initial_split_frac, random_state = self.get_seed())
        self.df_noise = self.df.drop(index=self.df_signal.index)
        self.df_noise.reset_index(inplace=True)
        del(self.df_noise['index'])
        self.df_signal.reset_index(inplace=True)
        del (self.df_signal['index'])

        signal_scores = self.cross_validate()
        signal_average_score = [np.mean(signal_scores)]

        frac_noise_list = []
        sigal_smoothed_score = []
        signal_iter_score = []

        window = 1

        frac_noisy_samples = 0.05
        frac_signal_samples = 0.01

        signal_gammas = []
        error_quantile = 0.90
        min_rel_error = 0.05

        # ===============================================================
        # Iterate
        # ===============================================================
        max_allowed_error = []
        for iter in range(self.max_iterations):

            # ========================
            # remove outliers from  signal
            # ========================
            new_score=  self.propose_sample_removal()

            if self.max_signal_error < self.min_mse:
                self.max_signal_error = self.min_mse

            gamma = signal_average_score[-1] / new_score
            signal_gammas.append(new_score)
            np.random.RandomState(self.get_seed())
            u = np.random.rand(1)
            signal_frac = len(self.df_signal) / len(self.df_noise)

            if u <= gamma:
                if signal_frac > self.min_signal_ratio:
                    new_score = self.damping(signal_average_score[-1], new_score, self.damping_weight)
                    self.df_noise = pd.concat([self.df_noise, self.df_s2o], axis=0)
                    self.df_noise.reset_index(inplace=True)
                    del (self.df_noise['index'])
                    self.df_signal = self.propose_df_signal
                    signal_average_score.append(new_score)
            else:
                signal_average_score.append(signal_average_score[-1])

            # ================================
            # remove signal from outlier
            # ================================
            yo_hat = self.signal_model.predict(self.df_noise[self.features])
            yo_true = self.df_noise[self.target]
            self.df_noise['err'] = np.power((yo_hat - yo_true), 2)


            ### *****************
            max_allowed_error.append(self.max_signal_error)
            level = scipy.stats.hmean(max_allowed_error)
            # propose samples to remove
            err = self.df_noise['err'].values
            #rel_err = np.abs((self.signal_model.predict(df_noise[self.features]) - df_noise[self.target])) / np.abs(df_noise[self.target])
            w = np.zeros_like(yo_true.values)
            if 1:
                ww = iter/1
                if ww> 1:
                    ww = 1.0
                w[err < ww*self.max_signal_error] = 1.0
                w = w / np.sum(w)
                w[np.isnan(w)] = 0
            else:
                w = 1 /err
                w = w / np.sum(w)

            nn = int(len(self.df_noise.index.values) * self.frac_signal_samples)
            if nn > len(w[w > 0]):
                nn = len(w[w > 0])
            if nn > 0:
                np.random.seed(self.get_seed())
                move_samples = choice(self.df_noise.index.values,
                                      nn,
                                      p=w,
                                      replace=False)
            else:
                move_samples = []

            df_move = self.df_noise.iloc[move_samples]
            df_noise_new = self.df_noise.drop(index=move_samples)
            df_noise_new.reset_index(inplace=True)
            del (df_noise_new['index'])

            df_signal_ = pd.concat([self.df_signal, df_move], axis=0)
            df_signal_.reset_index(inplace=True)
            del (df_signal_['index'])

            self.estimator.set_params(random_state=self.get_seed())
            self.estimator.set_params(seed=self.get_seed())
            signal_scores = cross_val_score(self.estimator, df_signal_[self.features], df_signal_[self.target],
                                            scoring=self.score, cv=self.kfolds)
            new_score = np.mean(signal_scores)

            np.random.RandomState(self.get_seed())
            u = np.random.rand(1)
            gamma = signal_average_score[-1] / new_score
            signal_frac = len(self.df_signal) / len(self.df_noise)

            if u <= gamma * window:

                if signal_frac < self.max_signal_ratio:
                    self.df_signal = df_signal_.copy()
                    signal_average_score.append(new_score)
                    del (df_noise_new['err'])
                    self.df_noise = df_noise_new.copy()

            else:
                del (self.df_noise['err'])
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

            N = self.max_iterations
            if iter == 0:
                columns = ['iter', 'score', 'sscore'] + list(range(len(self.df)))
                df_results = pd.DataFrame(np.nan, index=list(range(N)), columns=columns)
                df_results['iter'] = np.arange(N)
            iter_mask = df_results['iter'] == iter
            df_results.loc[iter_mask, 'score'] = signal_iter_score[-1]
            df_results.loc[iter_mask, 'sscore'] = sigal_smoothed_score[-1]
            signal_ids = self.df_signal['id'].values.tolist()
            df_results.loc[iter_mask, list(range(len(self.df)))] = 0
            df_results.loc[iter_mask, signal_ids] = 1

            if np.mod(iter, 100) == 0:
                sigs = self.df_signal.sample(frac=0.1, random_state=self.get_seed())
                noss = self.df_noise.sample(frac=0.1, random_state=self.get_seed())

                self.df_signal = self.df_signal.drop(sigs.index)
                self.df_signal = pd.concat([self.df_signal, noss])
                self.df_signal.reset_index(inplace=True)
                del (self.df_signal['index'])

                self.df_noise = self.df_noise.drop(noss.index)
                self.df_noise = pd.concat([self.df_noise, sigs])
                self.df_noise.reset_index(inplace=True)
                del (self.df_noise['index'])

            if 1:
                self.visulize(
                    fig=fig, axs=axs,
                    signal_average_scroe=signal_iter_score, sigal_smoothed_score=sigal_smoothed_score,
                    iter=iter, frac_noise_list=frac_noise_list
                )

        return df_results
        x = 1
