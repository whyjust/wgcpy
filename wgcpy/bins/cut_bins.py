#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
File Name：     cut_bins.py
Author :        weiguang
date：          2021/6/21
Description :
"""
from sklearn.tree import DecisionTreeClassifier
from .chi_merge import *
import warnings
warnings.filterwarnings("ignore")

TREE_UNDEFINED = -2

def interpolate_binning(data, var, special_attributes=None):
    '''
    插值分箱
    :param data: pd.DataFrame
    :param var: str, the var of dataframe
    :param special_attributes: list or tuple, remove values of Series
    :return: list, the bins of variable
    '''
    if special_attributes is None:
        special_attributes = []
    check_type(data=data, special_attributes=special_attributes)

    bin_s = data[var].loc[~data[var].isin(special_attributes)].sort_values()
    check_single_value(series=bin_s)
    if bin_s.isna().sum() > 0:
        raise ValueError(f"detect nan values in {var}")
    value_list = list(bin_s.value_counts().sort_index().index)
    cp = [(value_list[i] + value_list[i + 1]) / 2 for i in np.arange(len(value_list) - 1)]

    if not check_unique(cp):
        cp = sorted(list(set(cp)))
    return cp


def quantile_binning(data, var, max_interval=10, special_attributes=None):
    '''
    等频分箱
    :param data: pd.DataFrame
    :param var: str, the var of DataFrame
    :param max_interval: int, the max bins of Series
    :param special_attributes: list or tuple, remove values of Series
    :return: list, the bins of variable
    '''
    if special_attributes is None:
        special_attributes = []

    check_type(data=data, special_attributes=special_attributes)
    bin_s = data[var].loc[~data[var].isin(special_attributes)].sort_values()
    check_single_value(series=bin_s)
    if bin_s.isna().sum() > 0:
        raise ValueError(f"detect nan values in {var}")

    unique_c = len(bin_s.value_counts())
    if unique_c < max_interval:
        print(f"the unique of {var} less then {max_interval}")
        warnings.warn(f"warnings!",UserWarning)
        max_interval = unique_c
    cp = [bin_s.quantile(i) for i in np.linspace(0, 1, max_interval + 1)[1:-1]]

    if not check_unique(cp):
        cp = sorted(list(set(cp)))
    return cp


def distance_binning(data, var, max_interval=10, special_attributes=None):
    '''
    等距分箱
    :param data: pd.DataFrame
    :param var: str, the var of DataFrame
    :param max_interval: int, the max bins of Series
    :param special_attributes: list or tuple, remove values of Series
    :return: list, the bins of variable
    '''
    if special_attributes is None:
        special_attributes = []

    check_type(data=data, special_attributes=special_attributes)
    bin_s = data[var].loc[~data[var].isin(special_attributes)].sort_values()
    check_single_value(series=bin_s)
    if bin_s.isna().sum() > 0:
        raise ValueError(f"detect nan values in {var}")

    unique_c = len(bin_s.value_counts())
    if unique_c < max_interval:
        print(f"the unique of {var} less then {max_interval}")
        warnings.warn("warnings!", UserWarning)
        max_interval = unique_c
    cp = list(np.linspace(bin_s.min(), bin_s.max(), max_interval + 1, endpoint=True)[1:-1])

    if not check_unique(cp):
        cp = sorted(list(set(cp)))
    return cp


def mix_binning(data, var, max_interval=10, special_attributes=None):
    '''
    混合分箱
    :param data: pd.DataFrame
    :param var: str, the var of DataFrame
    :param max_interval: int, the max bins of Series
    :param special_attributes: list or tuple
    :return: list, the bins of variable
    '''
    if special_attributes is None:
        special_attributes = []

    check_type(data=data, special_attributes=special_attributes)
    bin_s = data[var].loc[~data[var].isin(special_attributes)].sort_values()
    if np.sum(bin_s.isna()) > 0:
        raise ValueError("detect nan values in {0}".format(var))

    unique_c = len(bin_s.value_counts())
    if unique_c < max_interval:
        print(f"the unique of {var} less then {max_interval}")
        warnings.warn("warnings!", UserWarning)
        max_interval = unique_c
    quantile_cp = [bin_s.quantile(i) for i in np.linspace(0,1,max_interval+1)][1:-1]
    distance_cp = list(np.linspace(quantile_cp[0], quantile_cp[-1], max_interval-1, endpoint=True)[1:-1])
    cp = [quantile_cp[0]] + distance_cp + [quantile_cp[-1]]

    if not check_unique(cp):
        cp = sorted(list(set(cp)))
    return cp


def tree_binning(data, var, target, special_attributes=None, tree_params=None):
    '''
    决策树分箱
    :param data: pd.DataFrame
    :param var: str, the var of dataframe
    :param target: the target of dataframe
    :param special_attributes: list or tuple, remove values of Series
    :param tree_params: dict, the params of DecisionTree
    :return: list, the bins of variable
    '''
    if tree_params is None:
        tree_params = {'max_leaf_nodes': 10, 'criterion': 'gini', 'min_samples_leaf': 0.01}

    if special_attributes is None:
        special_attributes = []
    check_type(data=data, special_attributes=special_attributes)
    bin_data = data[[var, target]].loc[~data[var].isin(special_attributes)]
    check_single_value(bin_data[var])
    if (np.sum(bin_data[var].isna()) > 0) or (np.sum(bin_data[target].isna()) > 0):
        raise ValueError(f"detect nan values in {var}")

    Dtree = DecisionTreeClassifier().set_params(**tree_params)
    Dtree.fit(bin_data[var].values.reshape(-1,1), bin_data[target].values.reshape(-1, 1))
    cp = sorted(Dtree.tree_.threshold[Dtree.tree_.threshold != TREE_UNDEFINED])
    if len(cp) == 0:
        raise ValueError(f"detect empty cp for {var} in tree bins!")
    return cp


def chi_binning(data, var, target, max_interval=10, special_attributes=None):
    '''
    卡方分箱
    :param data: pd.DataFrame
    :param var: str, the variable of dataframe
    :param target: str, the target of dataframe
    :param max_interval: int, the max interval of bins
    :param special_attributes: list or tuple
    :return: list, the cut points of variable
    '''
    if special_attributes is None:
        special_attributes = []
    check_type(data=data, special_attributes=special_attributes)
    binning_data = data[[var, target]].loc[~data[var].isin(special_attributes)]

    if (np.sum(binning_data[var].isna()) > 0) or (np.sum(binning_data[target].isna()) > 0):
        raise ValueError("detect nan values in {0}".format([var,target]))

    unique_c = len(binning_data[var].value_counts())
    if unique_c < max_interval:
        print(f"value_counts for {var} is {unique_c}, less than max_interval {max_interval}")
        warnings.warn("warnings!", UserWarning)
        max_interval = unique_c

    cp = cal_chi_merge(data, var, target, max_interval=max_interval, special_attributes=special_attributes)
    if (cp is np.nan) or not cp:
        raise ValueError("detect empty cp for {0} in chi_binning".format([var, target]))
    return cp

