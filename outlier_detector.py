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
                 target='y',
                 features=['x1'],
                 sample_id=None,
                 initial_split_frac=0.5,
                 max_iterations=400,
                 min_mse=None,
                 score='neg_mean_absolute_percentage_error',
                 kfolds=5,
                 test_frac=0.3,
                 max_signal_ratio=30,
                 min_signal_ratio=0.5,
                 cooling_rate=0.999,
                 damping_weight=0.8333,
                 estimator=None,
                 signal_error_quantile=0.80,
                 frac_noisy_samples=0.01,
                 frac_signal_samples=0.01,
                 ml_hyperparamters=None,
                 proposal_method = 'quantile',
                 leakage_rate = 1e-3,
                 symmetry_factor = 1.0

                 ):

        self.df = df.copy()
        self.initial_split_frac = initial_split_frac
        self.target = target
        self.features = features
        self.sample_id = sample_id
        self.symmetry_factor = symmetry_factor

        if sample_id is None:
            ids = range(len(df))
            if "sample_id" in df.columns:
                raise ValueError("The column 'sample_id' is already in the dataset;"
                                 " change the name and try again. ")
            self.sample_id = "sample_id"
            self.df[self.sample_id] = ids
        else:
            if np.any(self.df[self.sample_id].duplicated()):
                raise ValueError("Sample IDs must be unique")

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
        self.proposal_method = proposal_method
        self.leakage_rate = leakage_rate

        if ml_hyperparamters is None:
            self.ml_hyperparamters = {
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

    def xgb_estimator(self, params=None):
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

    def train_and_evaluate_signal_model(self):
        """

        :return:
        """
        if self.iter > 83:
            vvvv = 1
        df = self.df_signal.copy()
        df = df.sample(frac=1, random_state=self.get_seed()).reset_index(drop=True)
        train_df = df.sample(frac=(1 - self.test_frac), random_state=self.get_seed())
        test_df = df.drop(index=train_df.index)
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        # fit a model to flag possible samples to removing
        self.signal_model = self.xgb_estimator(params=self.ml_hyperparamters)
        self.signal_model.set_params(random_state=self.get_seed())
        self.signal_model.set_params(seed=self.get_seed())
        self.signal_model.fit(train_df[self.features], train_df[self.target])
        squar_error = np.power((self.signal_model.predict(test_df[self.features]) - test_df[self.target]), 2.0)
        return train_df, test_df, squar_error

    def quantile_proposal(self, train_df, test_df, signal_eval_square_error):
        """

        :param train_df:
        :param test_df:
        :param signal_eval_square_error:
        :return:
        """

        self.max_signal_error = np.quantile(signal_eval_square_error, self.signal_error_quantile)
        w = np.zeros_like(signal_eval_square_error)
        w[signal_eval_square_error >= self.max_signal_error ] = 1
        w[signal_eval_square_error< self.min_mse] = 0
        w = w / np.sum(w)
        w[np.isnan(w)] = 0

        if np.sum(w) == 0:
            proposed_signal_df = self.df_signal.copy()
            self.ids_s2o = []
        else:
            n_noisy_samples = int(len(test_df) * self.frac_noisy_samples)

            if n_noisy_samples > len(w[w > 0]):
                n_noisy_samples = len(w[w > 0])

            np.random.seed(self.get_seed())
            move_samples = choice(test_df[self.sample_id], n_noisy_samples, p=w, replace=False)
            df_to_move = test_df[test_df[self.sample_id].isin(move_samples)]
            self.ids_s2o = df_to_move[self.sample_id].values.tolist()

            mask2move = self.df_signal[self.sample_id].isin(self.ids_s2o )
            proposed_signal_df = self.df_signal[~mask2move]
            proposed_signal_df.reset_index(inplace=True)
            del (proposed_signal_df['index'])
        return proposed_signal_df

    def sig_mse_proposal(self, train_df, test_df, signal_eval_square_error):
        """

        :param train_df:
        :param test_df:
        :param signal_eval_square_error:
        :return:        """

        w = signal_eval_square_error.values
        w[w<self.min_mse] = 0
        if np.sum(w) == 0:
            w = w * 0.0
        else:
            w = w / np.sum(w)

        n_noisy_samples = int(len(test_df.index.values) * self.frac_noisy_samples)
        if n_noisy_samples > len(w[w > 0]):
            n_noisy_samples = len(w[w > 0])

        np.random.seed(self.get_seed())
        if np.sum(w) == 0:
            proposed_signal_df = self.df_signal.copy()
            self.ids_s2o = []
        else:
            np.random.seed(self.get_seed())
            move_samples = choice(test_df[self.sample_id], n_noisy_samples, p=w, replace=False)
            df_to_move = test_df[test_df[self.sample_id].isin(move_samples)]
            self.ids_s2o = df_to_move[self.sample_id].values.tolist()

            mask2move = self.df_signal[self.sample_id].isin(self.ids_s2o)
            proposed_signal_df = self.df_signal[~mask2move]
            proposed_signal_df.reset_index(inplace=True)
            del (proposed_signal_df['index'])

        return proposed_signal_df

    def propose_signal_change(self):

        method = 'random_walk'
        if method == 'random_walk':
            proposed_df_signal, proposed_df_noise = self.random_walk_proposal()
        elif method == "quantile":
            proposed_df_signal, proposed_df_noise = self.quantile_proposal()

    def sig_random_walk_proposal(self,train_df,test_df):
        """

        :param train_df:
        :param test_df:
        :return:
        """
        n_noisy_samples = int(len(test_df.index.values) * self.frac_noisy_samples)
        np.random.seed(self.get_seed())
        move_samples = choice(test_df[self.sample_id], n_noisy_samples, replace=False)
        df_to_move = test_df[test_df[self.sample_id].isin(move_samples)]
        self.ids_s2o = df_to_move[self.sample_id].values.tolist()

        mask2move = self.df_signal[self.sample_id].isin(self.ids_s2o)
        proposed_signal_df = self.df_signal[~mask2move]
        proposed_signal_df.reset_index(inplace=True)
        del (proposed_signal_df['index'])

        return proposed_signal_df



    def propose_sample_removal(self):
        """

        """
        train_df, test_df, signal_eval_square_error = self.train_and_evaluate_signal_model()

        if self.proposal_method in ['quantile']:
            proposed_signal_df = self.quantile_proposal(train_df,
                                                        test_df,
                                                        signal_eval_square_error)
        elif self.proposal_method in ['mse']:
            proposed_signal_df = self.sig_mse_proposal(train_df,
                                                       test_df,
                                                       signal_eval_square_error)
        elif self.proposal_method in ['random_walk']:
            proposed_signal_df = self.sig_random_walk_proposal(train_df,
                                                       test_df)

        else:
            raise ValueError("Unkown method")

        # random samples moves
        if self.proposal_method in ['quantile', 'mse']:
            leakage_rate = self.leakage_rate
            all_ids = set(self.df[self.sample_id])
            sig_ids = set(proposed_signal_df[self.sample_id])
            avail_ids = list(all_ids.difference(sig_ids))
            nn = int(leakage_rate * len(avail_ids))
            if nn == 0:
                nn = 1

            np.random.seed(self.get_seed())
            try:
                rand_samples = choice(avail_ids, size=nn, replace=False)
            except:
                vvvvv = 1
            new_sig_sample = list(set(rand_samples.tolist() + list(sig_ids)))
            new_sig_mask = self.df[self.sample_id].isin(new_sig_sample)
            proposed_signal_df = self.df[new_sig_mask]

        gb = self.xgb_estimator(params=self.ml_hyperparamters)
        gb.set_params(random_state=self.get_seed())
        gb.set_params(seed=self.get_seed())

        if 0: # old likelihood function
            signal_scores = cross_val_score(gb,
                                            proposed_signal_df[self.features],
                                            proposed_signal_df[self.target],
                                            scoring=self.score, cv=self.kfolds)
            new_score = np.mean(signal_scores)
        else:
            new_score = self.cross_validate_mse(gb,
                                    proposed_signal_df[self.features],
                                    proposed_signal_df[self.target],
                                    )



        self.propose_df_signal = proposed_signal_df
        all_ids = set(self.df[self.sample_id].values)
        sig_ids = set(self.propose_df_signal[self.sample_id].values)
        noise_ids = list(all_ids.difference(sig_ids))
        self.propose_df_noise = self.df[(self.df[self.sample_id].isin(noise_ids))].copy()



        return new_score

    def propose_sample_addition(self):
        """

        :return:
        """
        yo_hat = self.signal_model.predict(self.df_noise[self.features])
        yo_true = self.df_noise[self.target]
        self.df_noise['err'] = np.power((yo_hat - yo_true), 2)

        err = self.df_noise['err'].values
        w = np.zeros_like(yo_true.values)
        if self.proposal_method in ['quantile']:
            w[err <= self.max_signal_error] = 1.0
            if np.sum(w) > 0:
                w = w / np.sum(w)
            else:
                w = w * 0.0
        elif self.proposal_method in ['mse']:
            w = 1 / err
            w = w / np.sum(w)

        elif self.proposal_method in ['random_walk']:
            w = (1.0* np.ones_like(w))/len(w)



        nn = int(len(self.df_noise) * self.frac_signal_samples)
        if nn > len(w[w > 0]):
            nn = len(w[w > 0])
        if nn > 0:
            np.random.seed(self.get_seed())
            move_samples = choice(self.df_noise[self.sample_id],
                                  nn,
                                  p=w,
                                  replace=False)
        else:
            move_samples = []

        df_move = self.df_noise[self.df_noise[self.sample_id].isin(move_samples)]
        df_noise_new = self.df_noise[~(self.df_noise[self.sample_id].isin(move_samples))]

        if self.proposal_method in ['quantile', 'mse']:
            leakage_rate = self.leakage_rate
            all_ids = set(self.df[self.sample_id])
            noise_ids = set(df_noise_new[self.sample_id])
            avail_ids = list(all_ids.difference(noise_ids))
            nn = int(leakage_rate * len(avail_ids))
            if nn == 0:
                nn = 1

            np.random.seed(self.get_seed())
            rand_samples = choice(avail_ids, size=nn, replace=False)
            new_noise_sample = list(set(rand_samples.tolist() + list(noise_ids)))
            new_noise_mask = self.df[self.sample_id].isin(new_noise_sample)
            df_noise_new = self.df[new_noise_mask]
            propose_df_signal = self.df[~new_noise_mask]
        else:
            #all_ids = set(self.df[self.sample_id])
            noise_ids = set(df_noise_new[self.sample_id])
            #avail_ids = list(all_ids.difference(noise_ids))

            #np.random.seed(self.get_seed())
            #rand_samples = choice(avail_ids, size=nn, replace=False)
            #new_noise_sample = list(set(rand_samples.tolist() + list(noise_ids)))
            new_noise_mask = self.df[self.sample_id].isin(noise_ids)
            #df_noise_new = self.df[new_noise_mask]
            propose_df_signal = self.df[~new_noise_mask]




        # df_noise_new.reset_index(inplace=True)
        # del (df_noise_new['index'])

        # df_signal_ = pd.concat([self.df_signal, df_move], axis=0)
        # df_signal_.reset_index(inplace=True)
        # del (df_signal_['index'])

        self.propose_df_signal = propose_df_signal
        self.propose_df_noise = df_noise_new

        self.estimator.set_params(random_state=self.get_seed())
        self.estimator.set_params(seed=self.get_seed())
        cv = sklearn.model_selection.KFold(n_splits=self.kfolds, random_state=self.get_seed(),
                                           shuffle=True)  #

        if 0: # old likelihood function
            signal_scores = cross_val_score(self.estimator,
                                            self.propose_df_signal[self.features],
                                            self.propose_df_signal[self.target],
                                            scoring=self.score,
                                            cv=cv)
            new_score = np.mean(signal_scores)
        else:
            new_score = self.cross_validate_mse(self.estimator,
                                                self.propose_df_signal[self.features],
                                                self.propose_df_signal[self.target],
                                                )

        return new_score

    def damping(self, old_value, new_value, weight):
        score = (weight * old_value + (1 - weight) * new_value)
        return score

    def visulize(self, fig=None, axs=None,
                 signal_average_scroe=None, max_weight_change=None,
                 iter=0, frac_noise_list=None, acceptance_rate=None
                 ):
        if fig is None:
            fig, axs = plt.subplots(ncols=2, nrows=2)
            return fig, axs

        plt.ion()

        plot1, = axs[0][0].plot((np.abs(signal_average_scroe)), 'r', label='Error in Noise Model')
        plot2, = axs[0][1].plot(max_weight_change, 'b', label='Error in Signal Model')
        plot3, = axs[1][0].plot(frac_noise_list, 'b', label='Error in Signal Model')
        plot3, = axs[1][1].plot(acceptance_rate, 'b', label='Error in Signal Model')
        if iter == 0:
            axs[0][0].legend(loc='upper right', bbox_to_anchor=(0, 0.5))

        # plot1.set_ydata(np.log10(np.abs(signal_average_scroe)))
        # plot2.set_ydata(np.log10(np.abs(sigal_smoothed_score)))

        axs[0][0].set_title(str(iter))
        fig.canvas.draw()
        time.sleep(0.1)
        fig.canvas.flush_events()

        plt.tight_layout()
        plt.show()

    def cross_validate(self):

        estimator = self.estimator
        scoring = self.score

        df_ = self.get_signal_df()
        df_feat = df_[self.features]
        df_target = df_[self.target]
        cv = sklearn.model_selection.KFold(n_splits=self.kfolds, random_state=self.get_seed(),
                                           shuffle=True)  #

        estimator.set_params(random_state=self.get_seed())
        estimator.set_params(seed=self.get_seed())
        cv_score = cross_val_score(estimator,
                                   df_feat,
                                   df_target,
                                   scoring=scoring,
                                   cv=cv)

        return cv_score

    def cross_validate_mse(self, estimator, X, y):

        X_ = X.reset_index(drop=True)
        y_ = y.reset_index(drop = True)
        cv = sklearn.model_selection.KFold(n_splits=self.kfolds, random_state=self.get_seed(),
                                           shuffle=True)  #

        estimator.set_params(random_state=self.get_seed())
        estimator.set_params(seed=self.get_seed())


        sum_er_square = 0
        for train_index, val_index in cv.split(X_, y_):
            Xtrain, X_val = X_.loc[train_index], X_.loc[val_index]
            ytrain, y_val = y_.loc[train_index], y_.loc[val_index]

            estimator.fit(Xtrain, ytrain)
            pred = estimator.predict(X_val)
            sum_er_square = sum_er_square + (pred-y_val).pow(2.0).sum() # error per iteration

        err_var  = self.min_mse
        likelihood = (-0.5 * sum_er_square/err_var) + np.log(2.0 * np.pi * err_var)
        return likelihood

    def get_seed(self):
        seed = int(1e6 * self.RandomState.rand())
        return seed

    def get_signal_df(self):
        signal_df = self.df[self.df[self.sample_id].isin(self.signal_ids)]
        return signal_df.copy()

    def get_noise_df(self):
        noise_df = self.df[self.df[self.sample_id].isin(self.noise_ids)]
        noise_df = noise_df.copy()
        noise_df.reset_index(inplace=True)
        return noise_df

    def diagnose(self):

        N = self.max_iterations
        if self.iter == 0:
            ids_list = self.df[self.sample_id].values.tolist()
            columns = ['iter', 'score'] + ids_list
            self.df_results = pd.DataFrame(np.nan, index=list(range(N)), columns=columns)
            self.df_results['iter'] = np.arange(N)
        iter_mask = self.df_results['iter'] == self.iter
        self.df_results.loc[iter_mask, 'score'] = self.signal_iter_score[-1]
        signal_ids = self.df_signal[self.sample_id].values.tolist()
        self.df_results.loc[iter_mask, list(range(len(self.df)))] = 0
        self.df_results.loc[iter_mask, signal_ids] = 1

        sample_ids = self.df[self.sample_id].values
        sample_ids = sample_ids.tolist()
        if self.iter <= 10:
            current_weights = self.df_results[sample_ids]
            current_weights = current_weights.mean()
        else:
            last_10 = np.arange(self.iter - 10, self.iter + 1)
            last_10 = last_10.tolist()
            current_weights = self.df_results[self.df_results['iter'].isin(last_10)]
            current_weights = current_weights[sample_ids]
            current_weights = current_weights.mean()
        if self.iter > 0:
            dw = np.abs((self.previous_weights) - (current_weights))
            self.max_weight_change.append(np.max(dw))

        self.previous_weights = current_weights

        # if self.iter>20:
        #     acc_rate = np.mean(self.acceptance_flag[:-20])
        # else:
        acc_rate = np.mean(self.acceptance_flag)

        self.acceptance_rate.append(acc_rate)



    def purify(self, seed=123):
        """

        :param df:
        :param kwargs:
        :return:
        """
        self.switch = True

        ["iter", "number_signal_samples", "number_noise_samples", "score", "noise_signal_ratio"]
        fig, axs = self.visulize(fig=None)
        self.master_seed = seed
        self.RandomState = np.random.RandomState(seed)

        # ===============================================================
        # Shuffle, split, and initial cross validate
        # ===============================================================

        self.df = self.df.sample(frac=1, random_state=self.get_seed()).reset_index(drop=True)

        # todo: remove
        self.df_signal = self.df.sample(frac=self.initial_split_frac, random_state=self.get_seed())
        self.df_noise = self.df.drop(index=self.df_signal.index)

        self.signal_ids = self.df_signal[self.sample_id].values.tolist()
        self.noise_ids = self.df_noise[self.sample_id].values.tolist()

        self.df_noise.reset_index(inplace=True)
        del (self.df_noise['index'])
        self.df_signal.reset_index(inplace=True)
        del (self.df_signal['index'])

        if 0:
            signal_scores = self.cross_validate()
            signal_average_score = [np.mean(signal_scores)]
        else:
            signal_scores = self.cross_validate_mse(self.estimator,
                                                    self.df_signal[self.features],
                                                    self.df_signal[self.target])
            signal_average_score = [np.mean(signal_scores)]

        self.frac_noise_list = []
        self.signal_iter_score = []

        self.max_weight_change = []
        window = 1
        signal_gammas = []
        self.acceptance_rate = []
        self.acceptance_flag = []
        self.accept_count = 0
        window = self.symmetry_factor
        for iter in range(self.max_iterations):

            self.iter = iter

            # ========================
            # remove outliers from  signal
            # ========================

            new_score = self.propose_sample_removal()

            if self.proposal_method in ['quantile']:
                if self.max_signal_error < self.min_mse:
                    self.max_signal_error = self.min_mse

            gamma = signal_average_score[-1] / new_score
            signal_gammas.append(new_score)
            np.random.RandomState(self.get_seed())
            u = np.random.rand(1)
            signal_frac = len(self.df_signal) / len(self.df_noise)

            if u <= window * gamma:
                if signal_frac > self.min_signal_ratio:
                    #new_score = self.damping(signal_average_score[-1], new_score, self.damping_weight)
                    noise_ids = self.noise_ids + self.ids_s2o
                    self.noise_ids = noise_ids
                    self.df_noise = self.propose_df_noise

                    # this should be removed
                    self.df_noise.reset_index(inplace=True)
                    del (self.df_noise['index'])

                    self.df_signal = self.propose_df_signal
                    signal_average_score.append(new_score)
                    self.accept_count = self.accept_count + 1
                    self.acceptance_flag.append(1)
            else:
                signal_average_score.append(signal_average_score[-1])
                self.acceptance_flag.append(0)

            # ================================
            # remove signal from outlier
            # ================================
            new_score = self.propose_sample_addition()
            np.random.RandomState(self.get_seed())
            u = np.random.rand(1)
            gamma = signal_average_score[-1] / new_score
            signal_frac = len(self.df_signal) / len(self.df_noise)

            if u <= gamma * window:
                if signal_frac < self.max_signal_ratio:
                    self.df_signal = self.propose_df_signal.copy()
                    signal_average_score.append(new_score)
                    #del (self.propose_df_noise['err'])
                    self.df_noise = self.propose_df_noise.copy()
                    self.noise_ids = self.df_noise[self.sample_id].values.tolist()
                    self.accept_count = self.accept_count + 1
                    self.acceptance_flag.append(1)
            else:
                #del (self.df_noise['err'])
                signal_average_score.append(signal_average_score[-1])
                self.acceptance_flag.append(0)

            self.frac_noise_list.append(1.0 / signal_frac)
            if len(signal_average_score) > 1:
                ave_score = (signal_average_score[-1] + signal_average_score[-2]) / 2.0
                self.signal_iter_score.append(ave_score)
            else:
                self.signal_iter_score.append(signal_average_score[-1])



            self.diagnose()
            if np.mod(iter, 1) == 0:
                self.visulize(
                    fig=fig, axs=axs,
                    signal_average_scroe=self.signal_iter_score,
                    max_weight_change=self.max_weight_change,
                    iter=iter,
                    frac_noise_list=self.frac_noise_list,
                    acceptance_rate=self.acceptance_rate
                )

            print(">>> Iteration = {}, Score = {}".format(iter, ave_score))

        # compute average sample weight
        sample_ids = self.df[self.sample_id].values
        sample_ids = sample_ids.tolist()
        weights = self.df_results[sample_ids]
        self.mean_score = weights.mean()
        self.std_score = weights.std()
        self.std_score = self.std_score.reset_index().rename(columns={"index": self.sample_id, 0: "score_std"})
        self.mean_score = self.mean_score.reset_index().rename(columns={"index": self.sample_id, 0: "score_mean"})


        return self.df_results
