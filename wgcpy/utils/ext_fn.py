#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Project:       wgcpy
@File Name:     ext_fn.py
@Author:        weiguang
@Date:          2021/6/24
"""
import sys, os
import numpy as np
import pandas as pd
import time
from contextlib import contextmanager
import logging

PACKAGENAME = 'wgcpy'
LOGFILE = os.path.join(os.path.dirname(os.getcwd()), "info.log")

LEVEL = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'crit': logging.CRITICAL
}

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.info(f'[{name}] done in {time.time() - t0:.02f} s')

def init_logger(level=LEVEL['info']):
    logger = logging.getLogger(PACKAGENAME)
    logger.setLevel(level=level)
    if not logger.handlers:
        message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                           datefmt='%Y-%m-%d %H-%M-%S')
        ch_va = logging.StreamHandler(sys.stdout)
        ch_va.setLevel(logging.INFO)
        ch_va.setFormatter(fmt=message_format)
        logger.addHandler(ch_va)

        ch_lg = logging.FileHandler(LOGFILE, encoding='utf-8')
        ch_lg.setLevel(logging.INFO)
        ch_lg.setFormatter(fmt=message_format)
        logger.addHandler(ch_lg)
    return logger


logger = init_logger()
logger.info(f"wgcpy log save path: {LOGFILE}! ")

def is_monotonic(x):
    '''
    判断是否单调
    :param x: list or array or series
    :return: bool
    '''
    dx = np.diff(x)
    return np.all(dx < 0).astype(int) or np.all(dx > 0).astype(int)


def check_unique(x):
    '''
    检测是否唯一值
    :param x:  list or array or series
    :return: bool
    '''
    if len(np.unique(x)) != len(x):
        return False
    else:
        return True


def check_non_intersect(x, y):
    '''
    判断x与y是否存在交集
    :param x:  list or array or series
    :param y: list or array or series
    :return: bool
    '''
    if len(set(x) & set(y)) != 0:
        print("存在交集:%s" % (set(x) & set(y)))
        return False
    else:
        return True


def dict_reverse(d):
    '''
    将字典key与value反转
    :param d: dict
    :return: dict
    '''
    reverse_dict = {}
    for key, value in d.items():
        if value not in reverse_dict:
            reverse_dict[value] = [key]
        else:
            reverse_dict[value].append(key)
    return reverse_dict


def check_single_value(series):
    '''
    检测唯一值
    :param series: pd.Series
    :return: bool
    '''
    if len(series.value_counts()) == 1:
        raise ValueError("The value of Series is single!")


def check_type(data, special_attributes):
    '''
    检测数据类型
    :param data: pd.DataFrame
    :param special_attributes: list or tuple
    :return:
    '''
    if not isinstance(data, pd.DataFrame):
        raise Exception("the type of data must be DataFrame!")
    if not isinstance(special_attributes, (list, tuple)):
        raise Exception("the type of special_attribute must be list or tuple!")


def scale_range(input_array, scaled_min, scaled_max):
    '''
    区间缩放
    :param input_array: array
    :param scaled_min: float
    :param scaled_max: float
    :return: array
    '''
    input_array += -np.min(input_array)
    if scaled_max == scaled_min:
        raise Exception('scaled max equal scaled min, please check expected_array！')
    scaled_slope = np.max(input_array) * 1.0 / (scaled_max - scaled_min)
    input_array /= scaled_slope
    input_array += scaled_min
    return input_array


def generate_counts(arr, breakpoints):
    '''
    Generates counts for each bucket by using the bucket values
    :param arr: ndarray of actual values
    :param breakpoints: list of bucket values
    :return cnt_array: counts for elements in each bucket, length of breakpoints array minus one
    :return score_range_array: bins of cut
    '''

    def count_in_range(arr, low, high, start):
        '''
        Counts elements in array between low and high values)
        :param arr: ndarray of actual values
        :param low: float, the boundray of left bin
        :param high: float, the boundray of right bin
        :param start: bool, the mode of bins
        '''
        if start:
            cnt_in_range = len(np.where(np.logical_and(arr >= low, arr <= high))[0])
        else:
            cnt_in_range = len(np.where(np.logical_and(arr > low, arr <= high))[0])
        return cnt_in_range

    cnt_array = np.zeros(len(breakpoints) - 1)
    score_range_array = [''] * (len(breakpoints) - 1)
    for i in range(1, len(breakpoints)):
        cnt_array[i - 1] = count_in_range(arr, breakpoints[i - 1], breakpoints[i], i == 1)
        if 1 == i:
            score_range_array[i - 1] = '[' + str(round(breakpoints[i - 1], 4)) + ',' + str(
                round(breakpoints[i], 4)) + ']'
        else:
            score_range_array[i - 1] = '(' + str(round(breakpoints[i - 1], 4)) + ',' + str(
                round(breakpoints[i], 4)) + ']'
    return cnt_array, score_range_array


def sub_psi(e_perc, a_perc):
    '''
    cal psi value
    :param e_perc: array, the excepted array
    :param a_perc: array, the actual array
    :return: float, the psi value
    '''
    if a_perc == 0:
        a_perc = 0.00001
    if e_perc == 0:
        e_perc = 0.00001
    psi_v = (e_perc - a_perc) * np.log(e_perc * 1.0 / a_perc)
    return psi_v


class CheckDataType(object):
    def __init__(self, train_data, target_data):
        self.target_data = target_data
        self.train_data = train_data

    def _check_target_data_type(self):
        if not any([isinstance(self.target_data, obj_type) for obj_type in [pd.Series, np.ndarray, list]]):
            raise TypeError(
                '"target" must be "numpy.ndarray" or "Pandas.Series" or "list", got {} instead.'.format(type(self.target_data)))
        try:
            assert len(self.target_data.shape) == 1, 'target must be 1-D. It is {}-D instead.'.format(len(self.target_data.shape))
        except AttributeError:
            logger.info('Cannot determine shape of the {}. '
                        'Type must be "numpy.ndarray" or "Pandas.Series" or "list", got {} instead'.format(name,type(self.target_data)))

    @property
    def format_target(self):
        self._check_target_data_type()
        if isinstance(self.target_data, pd.Series):
            return self.target_data.values
        elif isinstance(self.target_data, np.ndarray):
            return self.target_data
        elif isinstance(self.target_data, list):
            return np.array(self.target_data)
        else:
            raise TypeError('"target" must be "numpy.ndarray" or "Pandas.Series" or "list", got {} instead.'.format(type(self.target_data)))

    @property
    def format_train(self):
        if isinstance(self.train_data, (pd.Series, pd.DataFrame)):
            return self.train_data
        else:
            raise TypeError("train data must be DataFrame or Series!")

