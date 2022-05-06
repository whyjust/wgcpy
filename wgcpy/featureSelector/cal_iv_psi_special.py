#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cal_iv_psi_special.py
@Time    :   2022/05/02 11:24:18
@Author  :   weiguang 
'''
import numpy as np
import pandas as pd

def equal_fre_cut(series, abnormal_value, bins):
    split_series = None
    bin_threshold = []

    series = series[pd.notnull(series)]
    series = series[~series.isin(abnormal_value)]
    series = series.sort_values()

    if len(series) > 0:
        # for i in range(bins, 1, -1):
        #     try:
        #         split_series = np.split(series, i)
        #         break
        #     except ValueError:
        #         pass

        if split_series is None:
            split_series = np.array_split(series, bins)

        for split in split_series:
            if len(split) == 0:
                continue
            else:
                bin_threshold.append(max(split))
            bin_threshold = [-np.inf] + bin_threshold
    else:
        split_series = None
        bin_threshold = None

    return split_series, bin_threshold
    #
    # fre = 1 / bins
    # cut_bins = list(np.arange(0, 1, fre)) + [1]
    # cut_bins = list(map(lambda x: round(x, 2), cut_bins))
    #
    # bin_threshold = series.quantile(cut_bins)
    # bin_threshold = sorted(set(bin_threshold.tolist()))
    #
    # return bin_threshold


def calculate_iv_woe(df, var, bin_size, abnormal_value_list, label):
    split_var_df, bin_threshold = equal_fre_cut(df[var], abnormal_value_list, bin_size)
    bin_threshold = sorted(set(bin_threshold))

    all_cnt = len(df)
    all_good = len(df[df[label] == 0])
    all_bad = len(df[df[label] == 1])

    bin_info = []
    index = 0
    df_var_null = df[pd.isnull(df[var])]

    # NAN单独分箱
    if len(df_var_null) > 0:
        bad_rate = len(df_var_null[df_var_null[label] == 1]) * 1.0 / len(df_var_null)
        good_per = len(df_var_null[df_var_null[label] == 0]) * 1.0 / all_good
        bad_per = len(df_var_null[df_var_null[label] == 1]) * 1.0 / all_bad
        bin_info.append([index, str(var), 'null_bin', len(df_var_null), len(df_var_null) * 1.0 / all_cnt, bad_rate,
                         good_per, bad_per])
        index += 1

    # 每一个异常值单独分箱
    for value in abnormal_value_list:
        abnormal_df = df[df[var] == value]
        if len(abnormal_df) > 0:
            bad_rate = len(abnormal_df[abnormal_df[label] == 1]) * 1.0 / len(abnormal_df)
            good_per = len(abnormal_df[abnormal_df[label] == 0]) * 1.0 / all_good
            bad_per = len(abnormal_df[abnormal_df[label] == 1]) * 1.0 / all_bad


            bin_info.append([index, str(var), str(value), len(abnormal_df), len(abnormal_df) * 1.0 / all_cnt, bad_rate,
                             good_per, bad_per])
            index += 1


    # 剩余值按bin_threshold分箱
    df = df[pd.notnull(df[var])]
    df = df[~df[var].isin(abnormal_value_list)]

    for i in range(1, len(bin_threshold)):
        mask = (df[var] > bin_threshold[i - 1]) & (df[var] <= bin_threshold[i])
        df1 = df[mask]

        if len(df1) > 0:
            bad_rate = len(df1[df1[label] == 1]) * 1.0 / len(df1)
            good_per = len(df1[df1[label] == 0]) * 1.0 / all_good
            bad_per = len(df1[df1[label] == 1]) * 1.0 / all_bad

            bin_info.append([index, str(var), '(' + str(bin_threshold[i-1]) + ', ' + str(bin_threshold[i]) + ']',
                             len(df1), len(df1) * 1.0 / all_cnt, bad_rate, good_per, bad_per])
            index += 1

    df_bin_info = pd.DataFrame(bin_info, columns=['index', 'VarName', 'bin', 'bin_cnt', 'bin_pct', 'bad_rate', 'good_pct', 'bad_pct'])
    df_bin_info['woe'] = df_bin_info.apply(lambda x: np.log(x.good_pct * 1.0 / (x.bad_pct + 1e-5)), axis=1)
    df_bin_info['iv'] = df_bin_info.apply(lambda x:  (x.good_pct - x.bad_pct) * np.log(x.good_pct * 1.0 / (x.bad_pct + 1e-5)), axis=1)
    df_bin_info['woe'] = df_bin_info['woe'].replace({np.inf: 0, -np.inf: 0})
    df_bin_info['iv'] = df_bin_info['iv'].replace({np.inf: 0, -np.inf: 0})
    iv_sum = sum(df_bin_info['iv'])
    return df_bin_info, iv_sum, df_bin_info['woe'].values

def calculate_psi(initial, new, var, abnormal_value_list, bin_size):
    split_var_df, bin_threshold = equal_fre_cut(initial[var], abnormal_value_list, bin_size)

    if split_var_df is None:
        return None, None

    bin_threshold = sorted(set(bin_threshold))
    initial_per = []
    new_per = []

    total_initial = len(initial)
    total_new = len(new)
    bin_info = []
    index = 0

    # NAN单独分箱
    initial_var_null = initial[pd.isnull(initial[var])]
    new_var_null = new[pd.isnull(new[var])]

    initial_null_len = len(initial_var_null)
    new_null_len = len(new_var_null)

    if initial_null_len > 0:
        initial_per.append(initial_null_len)
        new_per.append(new_null_len)
        bin_info.append([index, str(var), 'null_bin', initial_null_len, initial_null_len / total_initial,
                         new_null_len, new_null_len / total_new])
        index += 1


    # 每一个异常值单独分箱
    for value in abnormal_value_list:
        initial_abnormal_df = initial[initial[var] == value]
        new_abnormal_df = new[new[var] == value]

        initial_abnormal_len = len(initial_abnormal_df)
        new_abnormal_len = len(new_abnormal_df)

        if initial_abnormal_len > 0:
            initial_per.append(initial_abnormal_len)
            new_per.append(new_abnormal_len)

            bin_info.append([index, str(var), str(value), initial_abnormal_len, initial_abnormal_len / total_initial,
                             new_abnormal_len, new_abnormal_len / total_new])
            index += 1


    # 剩余值按bin_threshold分箱
    initial = initial[pd.notnull(initial[var])]
    initial = initial[~initial[var].isin(abnormal_value_list)]
    new = new[pd.notnull(new[var])]
    new = new[~new[var].isin(abnormal_value_list)]

    for i in range(1, len(bin_threshold)):
        initial_mask = (initial[var] > bin_threshold[i - 1]) & (initial[var] <= bin_threshold[i])
        initial_tmp_len = len(initial[initial_mask])

        new_mask = (new[var] > bin_threshold[i - 1]) & (new[var] <= bin_threshold[i])
        new_tmp_len = len(new[new_mask])

        if initial_tmp_len > 0:
            initial_per.append(initial_tmp_len)
            new_per.append(new_tmp_len)

            bin_info.append([index, str(var), '(' + str(bin_threshold[i-1]) + ', ' + str(bin_threshold[i]) + ']',
                             initial_tmp_len, initial_tmp_len / total_initial,
                             new_tmp_len, new_tmp_len / total_new])
            index += 1


    # 计算PSI
    initial_per = np.array(initial_per) / total_initial
    new_per = np.array(new_per) / total_new
    psi_value = np.sum([sub_psi(initial_per[i], new_per[i]) for i in range(len(initial_per))])
    df_bin_info = pd.DataFrame(bin_info, columns=['index', 'VarName', 'bin', 'expected_cnt', 'expected_pct', 'actucal_cnt', 'actucal_pct'])
    return psi_value, df_bin_info


def sub_psi(initial_per, new_per):
    if initial_per == 0:
        initial_per = 0.0001

    if new_per == 0:
        new_per = 0.0001

    sub_psi_value = (initial_per - new_per) * np.log(initial_per * 1.0 / new_per)
    return sub_psi_value
