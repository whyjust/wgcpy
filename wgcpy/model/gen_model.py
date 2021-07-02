#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Project:       wgcpy
@File Name:     gen_model.py
@Author:        weiguang
@Date:          2021/6/24
"""
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wgcpy.utils.ext_fn import *
from scipy.stats import ks_2samp
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, TimeSeriesSplit, LeaveOneGroupOut

class GenCVModel(CheckDataType):
    '''
    通过CV方式计算特征重要度
    '''
    def __init__(self, train_data, target_data):
        super(GenCVModel, self).__init__(train_data, target_data)
        self.target_data = self.format_target
        self.train_data = self.format_train
        self.n_splits = None
        self.kfold = None

    def _gen_kfold(self, kfold=None, groups=None):
        '''
        数据集切分
        :param kfold: str, the type of cv
        :param groups: ndarray or Series, the groups of split data, it is useful
                       when kfold is GroupKfold or LeaveOneGroupOut
        :return:
        '''
        if kfold == "GroupKfold":
            groups = self._check_groups(groups=groups)
            k_fold = GroupKFold(n_splits=self.n_splits)
            self.kfold = k_fold.split(self.train_data, self.target_data, groups=groups)

        elif kfold == "StratifiedKFold":
            k_fold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=10)
            self.kfold = k_fold.split(self.train_data, self.target_data)

        elif kfold == "KFold":
            k_fold = KFold(n_splits=self.n_splits, shuffle=True, random_state=10)
            self.kfold = k_fold.split(self.train_data)

        elif kfold == "TimeSeriesSplit":
            k_fold = TimeSeriesSplit(n_splits=self.n_splits)
            self.kfold = k_fold.split(self.train_data)

        elif kfold == "LeaveOneGroupOut":
            k_fold = LeaveOneGroupOut()
            groups = self._check_groups(groups=groups)
            self.kfold = k_fold.split(self.train_data, self.target_data, groups=groups)

        else:
            raise ValueError("cv need to be one of GroupKfold,StratifiedKFold,KFold,TimeSeriesSplit,LeaveOneGroupOut!")

    def _check_groups(self, groups):
        '''
        检验groups组数
        :param groups:
        :return:
        '''
        if groups.nunique() <= self.n_splits:
            self.n_splits = groups.nunique()
        return groups

    @staticmethod
    def get_ks_score(y_true, y_pred):
        d = zip(y_pred, y_true)
        bad = [k[0] for k in d if k[1] == 1]
        d = zip(y_pred, y_true)
        good = [k[0] for k in d if k[1] == 0]
        if len(bad) + len(good) < 100:
            return 0
        return ks_2samp(bad, good)[0]

    def cross_validation(self, params, kfold=None, groups=None, categorical_feature=None, n_splits=5):
        '''
        数据交叉验证
        :param params: dict, the params of estimator
        :param kfold: str, type of cv
        :param groups: ndarray or Series, the groups of split data
        :param categorical_feature: list or ndarray, the category feats of dataframe
        :param n_splits: int, the splits of data
        :return:
        '''
        if categorical_feature is None:
            categorical_feature = []

        if len(categorical_feature) > 0:
            self.train_data.loc[:, categorical_feature] = self.train_data.loc[:, categorical_feature].astype('category')

        self.n_splits = n_splits
        self._gen_kfold(kfold=kfold, groups=groups)

        oof = np.zeros(len(self.train_data))
        feature_importance_values = np.zeros(len(self.train_data.columns))
        valid_scores = []
        train_scores = []
        valid_ks_arr = []
        train_ks_arr = []

        for fold_, (trn_idx, val_idx) in enumerate(self.kfold):
            logger.info(f"start cv calulate, Fold: {fold_}")
            tr_x, tr_y = self.train_data.iloc[trn_idx, :], self.target_data[trn_idx]
            val_x, val_y = self.train_data.iloc[val_idx, :], self.target_data[val_idx]

            logger.info(f"train shape: {tr_x.shape}, train badRate: {tr_y.mean()}, "
                        f"valid shape: {val_x.shape}, valid badRate: {val_y.mean()}")

            model = lgb.LGBMClassifier().set_params(**params)

            if len(categorical_feature) == 0:
                model.fit(tr_x, tr_y, eval_metric='auc',
                          eval_set=[(val_x, val_y), (tr_x, tr_y)],
                          eval_names=['valid', 'train'],
                          early_stopping_rounds=100, verbose=100)
            else:
                model.fit(tr_x, tr_y, eval_metric='auc', categorical_feature=categorical_feature,
                          eval_set=[(val_x, val_y), (tr_x, tr_y)],eval_names=['valid', 'train'],
                          early_stopping_rounds=100, verbose=100)

            best_iteration = model.best_iteration_
            feature_importance_values += model.feature_importances_ / self.n_splits
            valid_prob = model.predict_proba(val_x, num_iteration=best_iteration)[:, 1]
            train_prob = model.predict_proba(tr_x, num_iteration=best_iteration)[:, 1]
            oof[val_idx] = valid_prob
            valid_score = model.best_score_['valid']['auc']
            train_score = model.best_score_['train']['auc']
            valid_ks = self.get_ks_score(val_y, valid_prob)
            train_ks = self.get_ks_score(tr_y, train_prob)

            valid_scores.append(valid_score)
            train_scores.append(train_score)
            valid_ks_arr.append(valid_ks)
            train_ks_arr.append(train_ks)

        del tr_x, val_x
        gc.collect()

        feats_importances = pd.DataFrame({'feature': self.train_data.columns, 'importance': feature_importance_values})
        feats_importances = feats_importances.sort_values('importance', ascending=False).reset_index()
        feats_importances['importance_normalized'] = feats_importances['importance'] / feats_importances['importance'].sum()
        feats_importances['cumsum'] = feats_importances['importance_normalized'].cumsum()
        valid_auc = roc_auc_score(self.target_data, oof)
        valid_scores.append(valid_auc)
        train_scores.append(np.mean(train_scores))
        valid_ks_arr.append(np.mean(valid_ks_arr))
        train_ks_arr.append(np.mean(train_ks_arr))

        fold_names = list(range(self.n_splits))
        fold_names.append('avg_score')
        result = pd.DataFrame({'fold': fold_names,
                               'train_auc': train_scores,
                               'valid_auc': valid_scores,
                               'train_ks': train_ks_arr,
                               'valid_ks': valid_ks_arr
                               })
        logger.info("cross validation complete!")
        return feats_importances, result


class IncreaseCVSelector(CheckDataType):
    def __init__(self, train_data, target_data):
        '''
        :param train_data: pd.Dataframe
        :param target_data: ndarray
        '''
        super(IncreaseCVSelector, self).__init__(train_data, target_data)
        self.feats_importances = None
        self.train_data = self.format_train
        self.target_data = self.format_target


    def _sort_feats_importanes(self, feats_importances):
        '''
        特征重要性排序
        :param feats_importances: pd.DataFrame
        :return:
        '''
        self.feats_importances = feats_importances.sort_values('importance', ascending=False)

    @staticmethod
    def _gen_cv_result(params, train_set, folds):
        '''
        cv模型初始化
        :param params: dict, the params of lgb
        :param train_set: DataSet, the Dataset of train
        :param folds: the type of split date
        :return: dict, the cv result
        '''
        cv_result = lgb.cv(params=params, train_set=train_set, folds=folds, shuffle=True)
        return cv_result

    def get_lgb_cv_score(self, feats_importances, total_iter, step, incre_params=None,
                         categorical_feature=None, auc_interval=None):
        '''
        模型通过cv自动筛选特征
        :param feats_importances: pd.Dataframe
        :param total_iter: int, the max iterator of traing round
        :param step: int, the numbers of add features
        :param incre_params: dict, the params of increase lgb model
        :param categorical_feature: list, the category of features
        :param auc_interval: int, the miniumn interval of auc increase
        :return:
            result: pd.DataFrame, the every step of cv result
            record_increase_feats: pd.DataFrame, the seleted of feats
        '''
        if categorical_feature is None:
            categorical_feature = []
        if auc_interval is None:
            auc_interval = 0.001
        if incre_params is None:
            incre_params = {
                "max_depth": 3,
                "learning_rate": 0.1,
                "num_boost_round": 600,
                "metrics": "auc",
                "verbose_eval": 200,
                "verbose":-1,
                "early_stopping_rounds": 100,
                "seed": 10
            }

        self._sort_feats_importanes(feats_importances=feats_importances)
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
        val_auc_arr = []
        val_auc_diff = []
        increase_feats = []
        feats_add_arr = []
        auc_initial = 0.5
        for i in range(0, min(len(self.feats_importances), total_iter), step):
            feats_add = self.feats_importances['feature'][i:i+step].values.tolist()
            logger.info(f"step:{i}, feature cnt:{len(increase_feats+feats_add)}")
            category_cols = [i for i in increase_feats+feats_add if i in categorical_feature]

            if len(category_cols) == 0:
                train_set = lgb.Dataset(self.train_data.loc[:, feats_add+increase_feats], label=self.target_data)
            else:
                train_set = lgb.Dataset(self.train_data.loc[:, feats_add+increase_feats], label=self.target_data,
                                        categorical_feature=category_cols)
            cv_result = self._gen_cv_result(params=incre_params, train_set=train_set, folds=folds)
            val_score = np.max(cv_result['auc-mean'])
            val_auc_arr.append(val_score)
            val_auc_diff.append(val_score - auc_initial)
            feats_add_arr.append(feats_add)

            if val_score - auc_initial < auc_interval:
                logger.info(f"delete the step features: {feats_add}, auc diff: {val_score - auc_initial}")
                continue

            increase_feats.extend(feats_add)
            auc_initial = val_score
            logger.info(f"add the step features, val_auc: {val_score}, val_std: {np.max(cv_result['auc-stdv'])}")
        result = pd.DataFrame(data={'step': range(step, min(len(self.feats_importances), total_iter)+step, step),
                                    'val_score': val_auc_arr, 'val_score_diff': val_auc_diff,
                                    'feats_add': feats_add_arr})
        record_increase_feats = self.feats_importances[self.feats_importances.feature.isin(increase_feats)]\
                                    .reset_index(drop=True)
        return result, record_increase_feats

