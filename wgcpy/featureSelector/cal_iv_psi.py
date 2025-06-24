#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Project:       wgcpy
@File Name:     cal_iv_psi.py
@Author:        weiguang
@Date:          2021/6/22
"""
from tqdm import tqdm
from wgcpy.utils.ext_fn import *
from wgcpy.bins.cut_bins import *

logger = init_logger()

def numeric_var_binning(data, var, target, bins, delta=1e-6):
    '''
    连续型变量计算iv值
    :param data: pd.DataFrame
    :param var: str, the variable of dataframe
    :param target: str, the target of dataframe
    :param bins: list, the bins of variable
    :param delta: float, the tiny offset of the divisor
    :return:
        bins_df: pd.DataFrame, the detail of bins
        IV: float, the value of IV
        bins: list, the cut bins of variable
    '''
    df = data[[var, target]].copy()
    df['bins'] = pd.cut(x=df[var], bins=bins)
    bins_df = pd.crosstab(df['bins'], df[target])
    bins_df.columns = ['num_0', 'num_1']
    bins_df['num_01'] = bins_df.apply(lambda x: sum(x), axis=1)
    bins_df['variable'] = var
    bins_df = bins_df[['variable', 'num_0', 'num_1', 'num_01']].reset_index()
    bins_df = bins_df.assign(
        pct_0_row=lambda x: x.num_0 / x.num_01,
        pct_1_row=lambda x: x.num_1 / x.num_01,
        pct_0_col=lambda x: x.num_0 / sum(x.num_0),
        pct_1_col=lambda x: x.num_1 / sum(x.num_1),
        pct_bin=lambda x: x.num_01 / sum(x.num_01)
    )
    bins_df['odds'] = bins_df['pct_1_row'] / bins_df['pct_0_row']
    bins_df['woe'] = np.log(bins_df['pct_1_col'] / (bins_df['pct_0_col'] + delta))
    bins_df['miv'] = (bins_df['pct_1_col'] - bins_df['pct_0_col']) * bins_df['woe']
    bins_df['woe'] = bins_df['woe'].replace({np.inf: 0, -np.inf: 0})
    bins_df['miv'] = bins_df['miv'].replace({np.inf: 0, -np.inf: 0})
    IV = bins_df['miv'].sum()
    bins_df['IV'] = [np.round(IV, 4)] * bins_df.shape[0]
    bins_df['is_monotonic'] = [is_monotonic(bins_df['woe'].values)] * bins_df.shape[0]
    return bins_df, IV, bins


def numeric_var_cal_iv(data, var, target, max_interval=10, method='DecisionTree', BRM=False,
                       special_attributes=None, tree_params=None, delta=1e-6):
    '''
    分箱并计算IV值
    :param data: pd.DataFrame
    :param var: str, the variable of dataframe
    :param target: str, the variable of dataframe
    :param max_interval: int
    :param method: the type of cut points
    :param BRM: bool, whether you need to merge according to monotony
    :param special_attributes: list or tuple, the special value of dataframe
    :param tree_params: dict, the params of DecisionTree
    :param delta: float, the tiny offset of the divisor
    :return:
        bins_df: pd.DataFrame, the detail of bins
        IV: float, the value of IV
        bins: list, the cut bins of variable
    '''
    if special_attributes is None:
        special_attributes = []

    df = data[[var, target]].copy()
    if method == 'chi':
        cp = chi_binning(df, var, target, max_interval, special_attributes)
    elif method == "tree":
        cp = tree_binning(df, var, target, special_attributes, tree_params)
    elif method == "distance":
        cp = distance_binning(df, var, max_interval, special_attributes)
    elif method == "quantile":
        cp = quantile_binning(df, var, max_interval, special_attributes)
    elif method == "mix":
        cp = mix_binning(df, var, max_interval, special_attributes)
    elif method == 'interpolate':
        cp = interpolate_binning(data, var, special_attributes)
    else:
        method_list = ['chi', 'tree', 'distance', 'quantile', 'mix', 'interpolate']
        raise ValueError(f"Can only use these method:{method_list}!")
    if BRM:
        cp = cutpoint_brm(data, var, target, cp, special_attributes)

    if len(special_attributes) > 1 or (len(special_attributes) == 1 and special_attributes[0] > df[var].min()):
        raise ValueError(f"special_attribute for {0} is {1}, not a empty list or a one element list contains the \
                            smallest value".format(var, special_attributes))
    bins = sorted([-np.inf] + special_attributes + cp + [np.inf])
    bins_df, IV, bins = numeric_var_binning(df, var, target, bins, delta)
    IV = np.round(IV, 4)
    logger.info(f"calculate IV: {var}, bins cnt:{len(bins)}, IV value: {IV}")
    return bins_df, IV, bins


def numeric_var_woe_transform(data, var, bins_df, bins):
    """
    数值型woe转换
    :param data: pd.DataFrame
    :param var: str, the variable of dataframe
    :param bins_df: pd.DataFrame, the detail of bins
    :param bins: list, the cut bins of variable
    :return: array, the woe array
    """
    df = data[[var]].copy()
    transform_dict = dict(zip(bins_df.index.tolist(), bins_df['woe'].values.tolsit()))
    df['bins'] = pd.cut(x=df[var], bins=bins)
    df['woe'] = df['bins'].replace(transform_dict)
    return df['woe'].values


def category_var_bins_merge(data, var, target, max_interval=10, method='default', special_attributes=None,
                            tree_params=None):
    '''
    将类别型特征分箱
    :param data: pd.DataFrame
    :param var: str, the variable of dataframe
    :param target: str, the target of dataframe
    :param max_interval: int, the max interval of variable
    :param method: str, the type of cut bins
    :param special_attributes: list or tuple
    :param tree_params: dict, the param of DecisionTree, useful when method is tree
    :return: dict, the bins dict of variable
    '''
    if special_attributes is None:
        special_attributes = []

    df = data[[var, target]][~data[var].isin(special_attributes)].copy()
    bins_df = pd.crosstab(df[var], df[target])
    bins_df.columns = ['num_0', 'num_1']
    bins_df['num_01'] = bins_df.apply(lambda x: sum(x), axis=1)
    bins_df = bins_df.assign(
        pct_0_row=lambda x: x.num_0 / x.num_01,
        pct_1_row=lambda x: x.num_1 / x.num_01
    )
    var_order = list(bins_df.sort_values(by="pct_1_row", ascending=False).index)
    stage1_bins_dict = {}
    for i in var_order:
        stage1_bins_dict[i] = var_order.index(i)
    df[var + '_stage1_bin'] = df[var].replace(stage1_bins_dict)
    stage2_bins_dict = {}
    if method == 'default':
        # 对于default方法 只需要把降序的对应关系稍加处理 把新分组的名称改成Bin_x
        for key in stage1_bins_dict.keys():
            stage2_bins_dict[key] = "Bin_" + "%03d" % (stage1_bins_dict.get(key))
    elif method == 'chi':
        # 对于ChiMerge方法 要把降序后的新分组当成变量传入到卡方分箱当中 然后把划分点拿出来对降序后的组号进行再分组
        cp = chi_binning(df, var + '_stage1_bin', target, max_interval)
        stage2_bins = sorted([-np.inf] + cp + [np.inf])
        stage1_bin_df = pd.DataFrame({'stage1_bin': np.arange(len(stage1_bins_dict))})
        stage1_bin_df['stage2_bin'] = pd.cut(x=stage1_bin_df['stage1_bin'], bins=stage2_bins)
        stage1_bin_df['stage2_bin_final'] = stage1_bin_df['stage2_bin']
        index = 0
        for i in stage1_bin_df['stage2_bin'].value_counts().sort_index().index:
            stage1_bin_df['stage2_bin_final'] = stage1_bin_df['stage2_bin_final'].replace({i: 'Bin_' + "%03d" % index})
            index += 1
        for i in stage1_bins_dict.keys():
            stage2_bins_dict[i] = stage1_bin_df['stage2_bin_final'][stage1_bin_df['stage1_bin']
                                                                    == stage1_bins_dict.get(i)].values[0]
    elif method == 'tree':
        # 对于DecisionTree方法，
        cp = tree_binning(df, var + '_stage1_bin', target, tree_params)
        stage2_bins = sorted([-np.inf] + cp + [np.inf])
        stage1_bin_df = pd.DataFrame({'stage1_bin': np.arange(len(stage1_bins_dict))})
        stage1_bin_df['stage2_bin'] = pd.cut(x=stage1_bin_df['stage1_bin'], bins=stage2_bins)
        stage1_bin_df['stage2_bin_final'] = stage1_bin_df['stage2_bin']
        index = 0
        for i in stage1_bin_df['stage2_bin'].value_counts().sort_index().index:
            stage1_bin_df['stage2_bin_final'] = stage1_bin_df['stage2_bin_final'].replace(
                {i: 'Bin_' + "%03d" % index})
            index += 1
        for i in stage1_bins_dict.keys():
            stage2_bins_dict[i] = stage1_bin_df['stage2_bin_final'][stage1_bin_df['stage1_bin']
                                                                    == stage1_bins_dict.get(i)].values[0]
    else:
        method_list = ['default', 'chi', 'tree']
        raise ValueError(f"Can only use these method:{method_list}!")

    for i in special_attributes:
        stage2_bins_dict[i] = "Bin_special"
    return stage2_bins_dict


def category_var_binning(data, var, target, bins, delta=1e-6):
    '''
    类别型特征IV详情
    :param data: pd.DataFrame
    :param var: str, the variable of dataframe
    :param target: str, the target of dataframe
    :param bins: dict, the bins dict of variable
    :param delta: float, the tiny offset of the divisor
    :return:
        bins_df: pd.DataFrame, the detail of bins
        IV: float, the value of IV
        bins: list, the cut bins of variable
    '''
    df = data[[var, target]].copy()
    reverse_bins = dict_reverse(bins)
    for i in reverse_bins.keys():
        reverse_bins[i] = ','.join(list(map(str, reverse_bins.get(i))))
    df['bins'] = df[var].replace(bins)
    bins_df = df[['bins', target]].groupby(['bins'], as_index=False).agg([np.sum, len])
    bins_df.columns = ['num_1', 'num_01']
    bins_df['num_0'] = bins_df['num_01'] - bins_df['num_1']
    bins_df['variable'] = var
    bins_df = bins_df[['variable', 'num_0', 'num_1', 'num_01']]
    bins_df = bins_df.assign(
        pct_0_row=lambda x: x.num_0 / x.num_01,
        pct_1_row=lambda x: x.num_1 / x.num_01,
        pct_0_col=lambda x: x.num_0 / sum(x.num_0),
        pct_1_col=lambda x: x.num_1 / sum(x.num_1),
        pct_bin=lambda x: x.num_01 / sum(x.num_01)
    )
    bins_df['odds'] = bins_df['pct_1_row'] / bins_df['pct_0_row']
    bins_df['woe'] = np.log(bins_df['pct_1_col'] / (bins_df['pct_0_col'] + delta))
    bins_df['miv'] = (bins_df['pct_1_col'] - bins_df['pct_0_col']) * bins_df['woe']
    bins_df['woe'] = bins_df['woe'].replace({np.inf: 0, -np.inf: 0})
    bins_df['miv'] = bins_df['miv'].replace({np.inf: 0, -np.inf: 0})
    IV = bins_df['miv'].sum()
    bins_df['IV'] = [np.round(IV, 4)] * bins_df.shape[0]
    bins_df['is_monotonic'] = [is_monotonic(bins_df['woe'].values)] * bins_df.shape[0]
    return bins_df, IV, bins


def category_var_cal_iv(data, var, target, max_interval=10, method='default', special_attributes=None,
                        tree_params=None, delta=1e-6):
    '''
    类别型特征计算IV
    :param data: pd.DataFrame
    :param var: str, the variable of dataframe
    :param target: str, the target of dataframe
    :param max_interval: int, the max interval of variable
    :param method: str, the type of cut bins
    :param special_attributes: list or tuple
    :param tree_params: dict, the param of DecisionTree
    :param delta: float, the tiny offset of the divisor
    :return:
        bins_df: pd.DataFrame, the detail of bins
        IV: float, the value of IV
        bins: dict, the cut bins of variable
    '''
    if special_attributes is None:
        special_attributes = []

    df = data[[var, target]].copy()
    bins = category_var_bins_merge(df, var, target, max_interval, method, special_attributes, tree_params)
    bins_df, IV, bins = category_var_binning(df, var, target, bins, delta)
    IV = np.round(IV, 4)
    logger.info(f"calculate IV: {var}, bins cnt:{len(bins)}, IV value: {IV}")
    return bins_df, IV, bins


def cal_total_var_iv(data, numeric_feats, category_feats, target,
                     max_interval=10, method='tree', BRM=False,
                     special_dict=None, tree_params=None):
    '''
    全量特征计算IV
    :param data: pd.DataFrame
    :param numeric_feats: list or ndarray, the numeric feats of data
    :param category_feats: list or ndarray, the category feats of data
    :param target: str, the target of data
    :param max_interval: int, the max interval of variable
    :param method: str, the methed of cut bins, default DecisionTree
    :param BRM: bool, whether to merge bins by monotone
    :param special_dict: dict, the special value of cut bins
    :param tree_params: dict, the tree param of DecisionTree, userful only when method='tree'
    :return: pd.DataFram, the details iv of total variable
    '''
    if special_dict is None:
        special_dict = {}

    iv_details = pd.DataFrame()
    if len(numeric_feats) > 0:
        for feat in tqdm(numeric_feats):
            num_bins_df, num_IV, num_bins = numeric_var_cal_iv(data, feat, target, max_interval=max_interval,
                                                               method=method, BRM=BRM, tree_params=tree_params,
                                                               special_attributes=special_dict.get(feat, None))
            num_bins_df['type'] = 'numeric'
            iv_details = pd.concat([iv_details, num_bins_df])
    if len(category_feats) > 0:
        for feat in tqdm(category_feats):
            cate_bins_df, cate_IV, cate_bins = category_var_cal_iv(data, feat, target, max_interval=max_interval,
                                                                   special_attributes=special_dict.get(feat, None),
                                                                   method=method, tree_params=tree_params)
            cate_bins_df['type'] = 'category'
            iv_details = pd.concat([iv_details, cate_bins_df])
    return iv_details


def category_var_woe_transform(data, var, bins_df, bins):
    '''
    类别型woe转换
    :param data: pd.DataFrame
    :param var: str, the variable of dataframe
    :param bins_df: pd.DataFrame, the detail of bins
    :param bins: dict, the bins dict of variable
    :return: array, the woe array
    '''
    df = data[[var]].copy()
    transform_dict = dict(zip(bins_df.index.tolist(), bins_df['woe'].values.tolist()))
    df['bins'] = df[var].replace(bins)
    df['woe'] = df['bins'].replace(transform_dict)
    return df['woe'].values


def numeric_var_cal_psi(expected_array, actual_array, bins=10, bucket_type='bins', detail=False, log=True):
    '''
    :param expected_array: numpy array of original values
    :param actual_array: numpy array of new values, same size as expected
    :param bins: number of percentile ranges to bucket the values into
    :param bucket_type: string, bins or quantiles for choose
    :param detail: bool, whether get the detail table
    :param log: bool, whether show the log info
    :return psi_value: psi
    '''
    expected_array = pd.Series(expected_array).dropna()
    actual_array = pd.Series(actual_array).dropna()
    if isinstance(list(expected_array)[0], str) or isinstance(list(actual_array)[0], str):
        raise Exception("the value of expected_array or actual_array need to be string!")

    if np.min(expected_array) == np.max(expected_array):
        return -1

    breakpoints = np.arange(0, bins + 1) / bins * 100
    if 'bins' == bucket_type:
        breakpoints = np.linspace(np.min(expected_array), np.max(expected_array), len(breakpoints))
    elif 'quantiles' == bucket_type:
        breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

    expected_cnt = generate_counts(expected_array, breakpoints)[0]
    expected_percents = expected_cnt / len(expected_array)
    actual_cnt = generate_counts(actual_array, breakpoints)[0]
    actual_percents = actual_cnt / len(actual_array)
    delta_percents = actual_percents - expected_percents
    score_range_array = generate_counts(expected_array, breakpoints)[1]
    sub_psi_array = [sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents))]

    if detail:
        psi_value = pd.DataFrame()
        psi_value['score_range'] = score_range_array
        psi_value['expecteds'] = expected_cnt
        psi_value['expected(%)'] = expected_percents * 100
        psi_value['actucals'] = actual_cnt
        psi_value['actucal(%)'] = actual_percents * 100
        psi_value['ac - ex(%)'] = delta_percents * 100
        psi_value['actucal(%)'] = psi_value['actucal(%)'].apply(lambda x: round(x, 4))
        psi_value['ac - ex(%)'] = psi_value['ac - ex(%)'].apply(lambda x: round(x, 4))
        psi_value['ln(ac/ex)'] = psi_value.apply(lambda row: np.log((row['actucal(%)'] + 0.00001)
                                                                    / (row['expected(%)'] + 0.00001)), axis=1)
        psi_value['psi'] = sub_psi_array
        psi_value = psi_value.append([{'score_range': 'summary',
                                       'expecteds': sum(expected_cnt),
                                       'expected(%)': 100,
                                       'actucals': sum(actual_cnt),
                                       'actucal(%)': 100,
                                       'ac - ex(%)': np.nan,
                                       'ln(ac/ex)': np.nan,
                                       'psi': np.sum(sub_psi_array)}], ignore_index=True)
        if log:
            logger.info(
                f"calclate psi complete, psi value: {psi_value.loc[psi_value.score_range == 'summary', 'psi'].values[0]}")
    else:
        psi_value = np.sum(sub_psi_array)
        if log:
            logger.info(f"calculate psi complete, psi value: {psi_value}")
    return psi_value
