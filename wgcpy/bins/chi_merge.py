#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
File Name：     chi_merge.py
Author :        weiguang
date：          2021/6/21
Description :
"""
import warnings
from utils.ext_fn import *

def split_data(df, col, num_split, special_attributes=None):
    """
    细粒度转粗粒度
    :param df: 按照col排序后的数据集
    :param col: 待分箱的变量
    :param num_split: 切分的组别数
    :param special_attributes: 在切分数据集的时候，某些特殊值需要排除在外
    :return: splitPoint: 初始化数据分箱的节点
    """
    if special_attributes is None:
        special_attributes = []
    df2 = df.copy()
    if len(special_attributes) > 0:
        df2 = df.loc[~df.col.isin(special_attributes)]

    N = df2.shape[0]
    n = int(N / num_split)
    splitPointIndex = [i * n for i in range(1, num_split)]
    rawValues = sorted(list(df2[col]))
    splitPoint = [rawValues[i] for i in splitPointIndex]
    splitPoint = sorted(list(set(splitPoint)))

    aa = pd.Series(splitPoint)
    if (aa[0] == 0.0) & (aa.shape[0] == 1):
        num_split = 1000
        n = int(N / num_split)
        splitPointIndex = [i * n for i in range(1, num_split)]
        rawValues = sorted(list(df2[col]))
        splitPoint = [rawValues[i] for i in splitPointIndex]
        splitPoint = sorted(list(set(splitPoint)))
    return splitPoint


def assign_group(x, bins):
    '''
    区间映射
    :param x: float or int
    :param bins: list or tuple, the cut bins
    :return: float or int, the value of
    '''
    N = len(bins)
    if x <= min(bins):
        return min(bins)
    elif x > max(bins):
        return max(bins)
    else:
        for i in range(N - 1):
            if bins[i] < x <= bins[i + 1]:
                return bins[i + 1]


def bin_bad_rate(df, col, target, grantRateIndicator=0):
    """
    用于计算col中数值对应的坏样本的占比情况
    :param df: 需要计算好坏比率的数据集
    :param col: 需要计算好坏比率的特征
    :param target: 好坏标签
    :param grantRateIndicator: 1返回总体的坏样本率，0不返回
    :return: 每箱的坏样本率，以及总体的坏样本率（当grantRateIndicator＝＝1时）
    """
    regroup = df.groupby(col, as_index=False).agg({target: ['count', 'sum']})
    regroup.columns = [col, 'total', 'bad']
    regroup = regroup.assign(bad_rate=lambda x: x.bad / x.total)
    dicts = dict(zip(regroup[col], regroup['bad_rate']))
    if grantRateIndicator == 0:
        return dicts, regroup
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    overallRate = B * 1.0 / N
    return dicts, regroup, overallRate


def cal_chi2(df, total_col, bad_col):
    '''
    计算卡方值
    :param df: pd.DataFrame, 包含全部样本总计与坏样本总计的数据框
    :param total_col: str, 全部样本的个数
    :param bad_col: str, 坏样本的个数
    :return: float, the chi2 value
    '''
    df2 = df.copy()
    badRate = sum(df2[bad_col]) * 1.0 / sum(df2[total_col])
    if badRate in [0, 1]:
        return 0
    df2 = df2.assign(good=lambda x: x.total - x.bad)
    goodRate = sum(df2['good']) * 1.0 / sum(df2[total_col])
    df2 = df2.assign(
        badExpected=lambda x: x.total * badRate,
        goodExpected=lambda x: x.total * goodRate
    )
    badCombined = zip(df2['badExpected'], df2[bad_col])
    goodCombined = zip(df2['goodExpected'], df2['good'])
    badChi = [(i[0] - i[1]) ** 2 / i[0] for i in badCombined]
    goodChi = [(i[0] - i[1]) ** 2 / i[0] for i in goodCombined]
    chi2 = sum(badChi) + sum(goodChi)
    return chi2


def assign_bin(x, cutOffPoints, special_attributes):
    '''
    bin分箱映射
    :param x: float
    :param cutOffPoints: list
    :param special_attributes: list
    :return: str
    '''
    cutOffPoints2 = [i for i in cutOffPoints if i not in special_attributes]
    numBin = len(cutOffPoints2)
    if x in special_attributes:
        i = special_attributes.index(x) + 1
        return 'Bin {}'.format(0 - i)
    if x <= cutOffPoints2[0]:
        return 'Bin 0'
    elif x > cutOffPoints2[-1]:
        return 'Bin {}'.format(numBin)
    else:
        for ii in range(0, numBin):
            if cutOffPoints2[ii] < x <= cutOffPoints2[ii + 1]:
                return 'Bin {}'.format(ii + 1)


def bad_rate_merge(df, col, cutOffPoints, target, special_attributes=None):
    '''
    将badRate0/1合并分箱节点
    :param df: pd.DataFrame, 包含全部样本总计与坏样本总计的数据框
    :param col: str, 计算好坏样本占比的列
    :param cutOffPoints: list, 分箱节点
    :param target: str, 目标变量
    :param special_attributes: list, 剔除的特殊值
    :return: list, 分箱节点
    '''
    if special_attributes is None:
        special_attributes = []
    while True:
        df2 = df.copy()
        df2['temp_bin'] = df2[col].apply(lambda x: assign_bin(x, cutOffPoints, special_attributes))
        binBadRate, regroup = bin_bad_rate(df2, 'temp_bin', target=target)
        indexForBad01 = regroup[regroup['bad_rate'].isin([0, 1])]['temp_bin'].tolist()
        if len(indexForBad01) == 0:
            return cutOffPoints

        bin = indexForBad01[0]
        # 移除最后一箱
        if bin == max(regroup['temp_bin']):
            cutOffPoints = cutOffPoints[:-1]
            if len(cutOffPoints) == 0:
                return np.nan
        # 移除第一箱
        elif bin == min(regroup['temp_bin']):
            cutOffPoints = cutOffPoints[1:]
            if len(cutOffPoints) == 0:
                return np.nan
        else:
            # 和前一箱进行合并，并且计算卡方值
            currentIndex = list(regroup['temp_bin']).index(bin)
            prevIndex = list(regroup['temp_bin'])[currentIndex - 1]
            df3 = df2.loc[df2['temp_bin'].isin([prevIndex, bin])]
            binBadRate, df2b = bin_bad_rate(df3, 'temp_bin', target)
            chisq1 = cal_chi2(df2b, 'total', 'bad')
            # 和后一箱进行合并，并且计算卡方值
            laterIndex = list(regroup['temp_bin'])[currentIndex + 1]
            df3b = df2.loc[df2['temp_bin'].isin([laterIndex, bin])]
            binBadRate, df2b = bin_bad_rate(df3b, 'temp_bin', target)
            chisq2 = cal_chi2(df2b, 'total', 'bad')
            if chisq1 < chisq2:
                cutOffPoints.remove(cutOffPoints[currentIndex - 1])
            else:
                cutOffPoints.remove(cutOffPoints[currentIndex])

        # 完成合并之后，需要再次计算新的分箱准则下，每箱是否同时包含好坏样本
        df2['temp_bin'] = df2[col].apply(lambda x: assign_bin(x, cutOffPoints, special_attributes))
        binBadRate, regroup = bin_bad_rate(df2, 'temp_bin', target)
        [minBadRate, maxBadRate] = [min(binBadRate.values()),
                                    max(binBadRate.values())]
        if minBadRate > 0 and maxBadRate < 1:
            break
    return cutOffPoints


def cal_chi_merge(df, col, target, max_interval, min_bin_pct=0.01, special_attributes=None):
    '''
    计算卡方分箱节点
    :param df: pd.DataFrame
    :param col: str, the variable of dataframe
    :param target: str, the target of dataframe
    :param max_interval: int, the max interval of variable
    :param min_bin_pct: float, the min percent of bin
    :param special_attributes: list or tuple, remove values of Series
    :return: list, the cutPoints of variable
    '''
    if special_attributes is None:
        special_attributes = []
    check_type(data=df, special_attributes=special_attributes)

    col_level = sorted(list(set(df[col])))
    N_distinct = len(col_level)
    if N_distinct <= max_interval:
        print(f"the number of {col} unique values less then {max_interval}!")
        warnings.warn("warnings!", UserWarning)
        return col_level[:-1]
    else:
        if len(special_attributes) >= 1:
            df2 = df[~df[col].isin(special_attributes)]
        else:
            df2 = df.copy()
        N_distinct = len(list(set(df2[col])))

        if N_distinct > 100:
            split_x = split_data(df2, col, num_split=100)
            df2['temp'] = df2[col].map(lambda x: assign_group(x=x, bins=split_x))
            if len(df2['temp'].unique()) == 1:
                return np.nan
        else:
            df2['temp'] = df2[col]

        binBadRate, regroup, overallRate = bin_bad_rate(df=df2, col='temp', target=target, grantRateIndicator=1)
        col_level = sorted(list(set(df2['temp'])))
        groupIntervals = [[i] for i in col_level]

        # 步骤：建立循环，不断合并最优的相邻两个组别，直到：
        # 1，最终分裂出来的分箱数<＝预设的最大分箱数
        # 2，每箱的占比不低于预设值（可选）
        # 3，每箱同时包含好坏样本
        # 如果有特殊属性，那么最终分裂出来的分箱数＝预设的最大分箱数－特殊属性的个数
        split_intervals = max_interval - len(special_attributes)
        if split_intervals == 1:
            return np.nan
        while len(groupIntervals) > split_intervals:
            chisqList = []
            for k in range(len(groupIntervals) - 1):
                temp_group = groupIntervals[k] + groupIntervals[k + 1]
                df2b = regroup.loc[regroup['temp'].isin(temp_group)]
                chisq = cal_chi2(df2b, 'total', 'bad')
                chisqList.append(chisq)
            best_comnbined = chisqList.index(min(chisqList))
            groupIntervals[best_comnbined] = groupIntervals[best_comnbined] + groupIntervals[best_comnbined + 1]
            groupIntervals.remove(groupIntervals[best_comnbined + 1])
        groupIntervals = [sorted(i) for i in groupIntervals]
        cutOffPoints = [max(i) for i in groupIntervals[:-1]]

        df2['temp_bin'] = df2['temp'].apply(lambda x: assign_bin(x=x, cutOffPoints=cutOffPoints,
                                                                 special_attributes=special_attributes))
        binBadRate, regroup = bin_bad_rate(df2, 'temp_bin', target)
        [minBadRate, maxBadRate] = [min(binBadRate.values()),
                                    max(binBadRate.values())]
        if minBadRate == 0 or maxBadRate == 1:
            cutOffPoints = bad_rate_merge(df, col, cutOffPoints, target)

        # 检查分箱后的最小占比
        if min_bin_pct > 0:
            groupValues = df2['temp'].apply(lambda x: assign_bin(x, cutOffPoints, special_attributes))
            df2['temp_bin'] = groupValues
            valueCounts = groupValues.value_counts().to_frame()
            N = sum(valueCounts['temp'])
            valueCounts['pcnt'] = valueCounts['temp'].apply(
                lambda x: x * 1.0 / N)
            valueCounts = valueCounts.sort_index()
            minPcnt = min(valueCounts['pcnt'])

            # 当最小箱占比不满足预设占比或者分箱节点数小于等于2时,合箱停止
            while minPcnt < min_bin_pct and len(cutOffPoints) > 2:
                indexForMinPcnt = valueCounts[valueCounts['pcnt'] == minPcnt].index.tolist()[0]
                # 占比最小箱是最后一箱
                if indexForMinPcnt == max(valueCounts.index):
                    cutOffPoints = cutOffPoints[:-1]
                # 占比最小箱是第一箱
                elif indexForMinPcnt == min(valueCounts.index):
                    cutOffPoints = cutOffPoints[1:]
                # 占比最小箱是中间箱
                else:
                    # 和前一箱合并计算卡方值
                    currentIndex = list(valueCounts.index).index(indexForMinPcnt)
                    prevIndex = list(valueCounts.index)[currentIndex - 1]
                    df3 = df2.loc[df2['temp_bin'].isin([prevIndex, indexForMinPcnt])]
                    binBadRate, df2b = bin_bad_rate(df3, 'temp_bin', target)
                    chisq1 = cal_chi2(df2b, 'total', 'bad')
                    # 和后一箱合并计算卡方值
                    laterIndex = list(valueCounts.index)[currentIndex + 1]
                    df3b = df2.loc[df2['temp_bin'].isin([laterIndex, indexForMinPcnt])]
                    binBadRate, df2b = bin_bad_rate(df3b, 'temp_bin', target)
                    chisq2 = cal_chi2(df2b, 'total', 'bad')
                    # 比较卡方值, 将较小的卡方值进行合箱
                    if chisq1 < chisq2:
                        cutOffPoints.remove(cutOffPoints[currentIndex - 1])
                    else:
                        cutOffPoints.remove(cutOffPoints[currentIndex])
                df2['temp_bin'] = df2['temp'].apply(lambda x: assign_bin(x, cutOffPoints, special_attributes))
                valueCounts = df2['temp_bin'].value_counts().to_frame()
                valueCounts['pcnt'] = valueCounts['temp'].apply(lambda x: x * 1.0 / N)
                valueCounts = valueCounts.sort_index()
                minPcnt = min(valueCounts['pcnt'])

            if cutOffPoints is np.nan:
                return np.nan
            return cutOffPoints


'''
-----------------------------
单调性合并相关函数
------------------------------
'''

def feature_monotone(x):
    '''
    判断不满足单调性的值对应index
    :param x: list, the cut of bins
    :return: dict
    '''
    monotone = [(x[i] < x[i + 1]) and (x[i] < x[i - 1]) or (x[i] > x[i + 1]) and (x[i] > x[i - 1]) for i in
                range(1, len(x) - 1)]
    index_of_monotone = [i + 1 for i in range(len(monotone)) if monotone[i]]
    return {'count_of_nonmonotone': monotone.count(True),
            'index_of_nonmonotone': index_of_monotone}


def bad_rate_monotone(df, sortByVar, target, special_attributes=None):
    '''
    判断某变量的坏账率是否是单调
    :param df: pd.DataFrame
    :param sortByVar: str
    :param target: str, the target of dataframe
    :param special_attributes: list or tuple
    :return: bool
    '''
    if special_attributes is None:
        special_attributes = []

    df2 = df.loc[~df[sortByVar].isin(special_attributes)]
    if len(set(df2[sortByVar])) <= 2:
        return True
    regroup = bin_bad_rate(df2, sortByVar, target)[1]
    combined = zip(regroup['total'], regroup['bad'])
    badRate = [x[1] * 1.0 / x[0] for x in combined]
    badRateNotMonotone = feature_monotone(badRate)['count_of_nonmonotone']
    if badRateNotMonotone > 0:
        return False
    else:
        return True


def monotone_merge(df, target, col):
    def merge_maxtrix(m, i, j, k):
        """
        :param i: 合并第i行
        :param m: 需要合并行的矩阵
        :param j: 合并第i行
        :param k: 删除第k行
        :return: 合并后的矩阵
        """
        m[i, :] = m[i, :] + m[j, :]
        m = np.delete(m, k, axis=0)
        return m

    def merge_adjacent_rows(i, bad_by_bin_current, bins_list_current, not_monotone_count_current):
        """
        :param i: 需要将第i行与前、后的行分别进行合并，比较哪种合并方案最佳。判断准则是，合并后非单调性程度减轻，且更加均匀
        :param bad_by_bin_current:合并前的分箱矩阵，包括每一箱的样本个数、坏样本个数和坏样本率
        :param bins_list_current: 合并前的分箱方案
        :param not_monotone_count_current:合并前的非单调性元素个数
        :return:分箱后的分箱矩阵、分箱方案、非单调性元素个数和衡量均匀性的指标balance
        """
        i_prev = i - 1
        i_next = i + 1
        bins_list = bins_list_current.copy()
        bad_by_bin = bad_by_bin_current.copy()
        not_monotone_count = not_monotone_count_current

        # 合并方案a：将第i箱与前一箱进行合并
        bad_by_bin2a = merge_maxtrix(bad_by_bin.copy(), i_prev, i, i)
        bad_by_bin2a[i_prev, -1] = bad_by_bin2a[i_prev, -2] / bad_by_bin2a[i_prev, -3]
        not_monotone_count2a = feature_monotone(bad_by_bin2a[:, -1])['count_of_nonmonotone']

        # 合并方案b：将第i行与后一行进行合并
        bad_by_bin2b = merge_maxtrix(bad_by_bin.copy(), i, i_next, i_next)
        bad_by_bin2b[i, -1] = bad_by_bin2b[i, -2] / bad_by_bin2b[i, -3]
        not_monotone_count2b = feature_monotone(bad_by_bin2b[:, -1])['count_of_nonmonotone']
        balance = ((bad_by_bin[:, 1] / N).T * (bad_by_bin[:, 1] / N))[0, 0]
        balance_a = ((bad_by_bin2a[:, 1] / N).T * (bad_by_bin2a[:, 1] / N))[0, 0]
        balance_b = ((bad_by_bin2b[:, 1] / N).T * (bad_by_bin2b[:, 1] / N))[0, 0]

        # 满足下述2种情况时返回方案a：（1）方案a能减轻非单调性而方案b不能；
        # （2）方案a和b都能减轻非单调性，但是方案a的样本均匀性优于方案b
        if (not_monotone_count2a < not_monotone_count_current) and \
                (not_monotone_count2b >= not_monotone_count_current) or \
                (not_monotone_count2a < not_monotone_count_current) and \
                (not_monotone_count2b < not_monotone_count_current) and \
                (balance_a < balance_b):
            bins_list[i_prev] = bins_list[i_prev] + bins_list[i]
            bins_list.remove(bins_list[i])
            bad_by_bin = bad_by_bin2a
            not_monotone_count = not_monotone_count2a
            balance = balance_a
        # 同样地，满足下述2种情况时返回方案b：
        # （1）方案b能减轻非单调性而方案a不能；
        # （2）方案a和b都能减轻非单调性，但是方案b的样本均匀性优于方案a
        elif (not_monotone_count2a >= not_monotone_count_current) and \
                (not_monotone_count2b < not_monotone_count_current) or \
                (not_monotone_count2a < not_monotone_count_current) and \
                (not_monotone_count2b < not_monotone_count_current) and \
                (balance_a > balance_b):
            bins_list[i] = bins_list[i] + bins_list[i_next]
            bins_list.remove(bins_list[i_next])
            bad_by_bin = bad_by_bin2b
            not_monotone_count = not_monotone_count2b
            balance = balance_b
        # 如果方案a和b都不能减轻非单调性，返回均匀性更优的合并方案
        else:
            if balance_a < balance_b:
                bins_list[i] = bins_list[i] + bins_list[i_next]
                bins_list.remove(bins_list[i_next])
                bad_by_bin = bad_by_bin2b
                not_monotone_count = not_monotone_count2b
                balance = balance_b
            else:
                bins_list[i] = bins_list[i] + bins_list[i_next]
                bins_list.remove(bins_list[i_next])
                bad_by_bin = bad_by_bin2b
                not_monotone_count = not_monotone_count2b
                balance = balance_b
        return {'bins_list': bins_list, 'bad_by_bin': bad_by_bin,
                'not_monotone_count': not_monotone_count,
                'balance': balance}

    N = df.shape[0]
    badrate_bin, bad_by_bin = bin_bad_rate(df=df, col=col, target=target)
    bins = list(bad_by_bin[col])
    bins_list = [[i] for i in bins]
    badRate = sorted(badrate_bin.items(), key=lambda x: x[0])
    badRate = [i[1] for i in badRate]
    not_monotone_count, not_monotone_position = feature_monotone(badRate)['count_of_nonmonotone'], \
                                                feature_monotone(badRate)['index_of_nonmonotone']
    # 迭代地寻找最优合并方案，终止条件是:当前的坏样本率已经单调，或者当前只有2箱
    while not_monotone_count > 0 and len(bins_list) > 2:
        all_possible_merging = []
        for i in not_monotone_position:
            merge_adjacent_rows = merge_adjacent_rows(
                i, np.mat(bad_by_bin), bins_list, not_monotone_count)
            all_possible_merging.append(merge_adjacent_rows)
        balance_list = [i['balance'] for i in all_possible_merging]
        not_monotone_count_new = [i['not_monotone_count'] for i in all_possible_merging]
        # 如果所有的合并方案都不能减轻当前的非单调性，就选择更加均匀的合并方案
        if min(not_monotone_count_new) >= not_monotone_count:
            best_merging_position = balance_list.index(min(balance_list))
        # 如果有多个合并方案都能减轻当前的非单调性，也选择更加均匀的合并方案
        else:
            better_merging_index = [
                i for i in range(len(not_monotone_count_new))
                if not_monotone_count_new[i] < not_monotone_count
            ]
            better_balance = [balance_list[i] for i in better_merging_index]
            best_balance_index = better_balance.index(min(better_balance))
            best_merging_position = better_merging_index[best_balance_index]
        bins_list = all_possible_merging[best_merging_position]['bins_list']
        bad_by_bin = all_possible_merging[best_merging_position]['bad_by_bin']
        not_monotone_count = all_possible_merging[best_merging_position]['not_monotone_count']
        not_monotone_position = FeatureMonotone(bad_by_bin[:, 3])['index_of_nonmonotone']
    return bins_list


def cutpoint_brm(data, var, label, cp, special_attributes=None):
    '''
    检验每一箱中坏样本分布是否单调，不单调需要将分箱进行上下合并
    :param data: pd.DataFrame
    :param var: str, the variable of dataframe
    :param label: str, the label of dataframe
    :param cp: list, the cut bins of variable
    :param special_attributes: list r tuple
    :return: list
    '''
    data = data.copy()
    var_cutoff = dict()
    col1 = str(var) + '_Bin'
    cp = sorted(list(cp))
    if special_attributes is not None:
        special_attributes = []
    # 将col1按照cp进行分组映射
    data[col1] = data[var].map(
        lambda x: assign_bin(
            x, cp, special_attributes=special_attributes
        )
    )
    binBadRate, regroup = bin_bad_rate(data, col1, label)
    [minBadRate, maxBadRate] = [min(binBadRate.values()),
                                max(binBadRate.values())]
    if minBadRate == 0 or maxBadRate == 1:
        if len(binBadRate) == 2:
            return np.nan
        else:
            # 坏账率分箱后检验与合箱
            cp = bad_rate_merge(data, var, cp, label)
            return cp
    data[col1] = data[var].map(
        lambda x: assign_bin(x, cp, special_attributes=special_attributes)
    )
    var_cutoff[var] = cp
    brm = bad_rate_monotone(data, col1, label, special_attributes=special_attributes)
    # 分箱后坏账率不单调，则继续进行合箱
    if not brm:
        bin_merge = monotone_merge(data, label, col1)
        removed_index = []
        for bin in bin_merge:
            if len(bin) > 1:
                indices = [int(b.replace('Bin ', '')) for b in bin]
                removed_index = removed_index + indices[0:-1]
        removed_point = [cp[k] for k in removed_index]
        for p in removed_point:
            cp.remove(p)
        var_cutoff[var] = cp
        data[col1] = data[var].map(
            lambda x: assign_bin(
                x, cp, special_attributes=special_attributes
            )
        )
    return cp
