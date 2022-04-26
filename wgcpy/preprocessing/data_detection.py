#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
File Name：     data_dectection.py
Author :        weiguang
date：          2021/6/18
Description :
"""
from utils.ext_fn import *
import os

logger = init_logger()

class DetectDF:
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            Exception("the type of df must be DataFrame!")
        self.df = df

    @staticmethod
    def _get_top_values(series, top=5, reverse=False):
        '''
        获取Top对应的值
        :param series: pd.Series
        :param top: int
        :param reverse: bool
        :return: pd.Series
        '''
        itype = 'top'
        counts = series.value_counts()
        counts = list(zip(counts.index, counts, counts.divide(series.size)))

        if reverse:
            counts.reverse()
            itype = 'bottom'

        template = "{0[0]}:{0[2]:.2%}"
        indexs = [itype + str(i + 1) for i in range(top)]
        values = [template.format(counts[i]) if i < len(counts) else None for i in range(top)]
        return pd.Series(values, index=indexs)

    @staticmethod
    def _get_describe(series, percentiles=None):
        '''
        获取数据表描述性统计结果
        :param series: pd.Series
        :param percentiles: list or ndarray
        :return: pd.DataFrame
        '''
        if percentiles is None:
            percentiles = [0.25, 0.5, 0.75]
        d = series.describe(percentiles)
        return d.drop('count')

    @staticmethod
    def _cal_na(series, special_value_dict=None):
        '''
        计算列的缺失占比
        :param series: pd.Series
        :param special_value_dict:
        :return: tuple
        '''
        if not special_value_dict:
            series = series.replace(special_value_dict)
        n = series.isnull().sum()
        return n, "{0:.2%}".format(n / series.size)

    @staticmethod
    def _is_numeric(series):
        '''
        数值型判断
        :param series: pd.Series
        :return: bool
        '''
        return series.dtype.kind in 'ifc'

    def detect(self, special_value_dict=None, output=None):
        '''
        数据表探查
        :param special_value_dict: dict
        :param output, str, the dir of save result
        :return: pd.DataFrame
        '''
        if special_value_dict is None:
            special_value_dict = {}
        if not isinstance(special_value_dict, dict):
            raise Exception("the type of the special_value_dict must be dict!")

        logger.info("start detect dataframe >>>>>>>>>")
        rows = []
        for name, series in self.df.items():
            numeric_index = ['mean', 'std', 'min', '1%', '10%', '50%', '75%', '90%', '99%', 'max']
            discrete_index = ['top1', 'top2', 'top3', 'top4', 'top5', 'bottom5', 'bottom4', 'bottom3', 'bottom2',
                              'bottom1']
            details_index = [numeric_index[i] + '_or_' + discrete_index[i] for i in range(len(numeric_index))]

            if self._is_numeric(series=series):
                des = self._get_describe(
                    series=series,
                    percentiles=[.01, .1, .5, .75, .9, .99]
                )
                des = des.tolist()
            else:
                top5 = self._get_top_values(series=series, top=5)
                bottom5 = self._get_top_values(series=series, top=5, reverse=True)
                des = top5.tolist() + bottom5[::-1].tolist()

            na_cnt, na_pct = self._cal_na(series=series, special_value_dict=special_value_dict)
            row = pd.Series(
                index=['type', 'size', 'missing', 'unique'] + details_index,
                data=[series.dtype, series.size, na_pct, series.nunique()] + des
            )
            row.name = name
            rows.append(row)
        des_df = pd.DataFrame(rows)
        if output:
            if not os.path.isdir(output):
                raise ValueError("output must be dir!")
            des_df.to_csv(os.path.join(output, 'des_df.csv'))
        return des_df

# @timecount()
# def describe_df(df, special_value_dict=None, plot=False):
#     '''
#     数据统计性描述
#     :param df: DataFrame
#     :param special_value_dict: dict
#     :param plot: bool
#     :return:
#     '''
#     df_copy = df.copy()
#     if not isinstance(df, pd.DataFrame):
#         Exception("df must be dataframe!")
#     if not isinstance(special_value_dict, dict):
#         Exception("special_value_dict must be dict!")
#     if special_value_dict:
#         df_copy = df_copy.replace(special_value_dict)
#
#     df_des = pd.concat([
#         (df_copy.shape[0] - df_copy.count()).to_frame('na_count'),
#         (1 - df_copy.count() / df_copy.shape[0]).to_frame('na_rate'),
#         df_copy.dtypes.to_frame('col_type'),
#         pd.DataFrame({
#             'unique_num': [len(df_copy[col].unique()) for col in df_copy]
#         }, index=df_copy.columns)
#     ], axis=1, sort=False)
#     df_des_detail = df_copy.describe().round(2).T
#
#     if plot:
#         plt.plot(range(len(df_des.index)),df_des['na_rate'].sort_values())
#         plt.title("Na Rate")
#         plt.show()
#
#     del df_copy
#     gc.collect()
#     return df_des,df_des_detail
