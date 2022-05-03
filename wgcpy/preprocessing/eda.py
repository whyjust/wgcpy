#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Project:       wgcpy
@File Name:     eda.py
@Author:        weiguang
@Date:          2021/6/18
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_boxplot(df, feats, figsize=(40,12), sub_col=2):
    '''
    绘制箱型图
    :param df: DataFrame
    :param feats: list or ndarray
    :param figsize: tuple
    :param sub_col: int
    :return:
    '''
    if not isinstance(df, pd.DataFrame):
        raise Exception("The type of df must be DataFrame")
    plt.subplots(figsize=figsize)

    for i,feat in enumerate(feats):
        ax1 = plt.subplot(sub_col, int(np.ceil(len(feats)/sub_col)), i+1)
        plt.xlim(df[feat].min(), df[feat].max()*1.1)
        sns.boxplot(x=df[feat], ax=ax1)
    plt.show()


def plot_feature_distribution(df, feats, label='flag', sub_col=5, figsize=(35,20)):
    '''
    绘制特征分布图
    :param df: DataFrame
    :param feats: list or ndarray
    :param label: string, the label of df
    :param sub_col: int
    :param figsize: tuple
    :return:
    '''
    if not isinstance(df, pd.DataFrame):
        raise Exception("The type of df must be DataFrame")
    label_v = np.unique(df[label])
    d1 = df[df[label].isin([label_v[1]])]
    d0 = df[df[label].isin([label_v[0]])]
    plt.figure(figsize=figsize)

    for i,feat in enumerate(feats):
        ax1 = plt.subplot(sub_col, int(np.ceil(len(feats)/sub_col)), i+1)
        sns.kdeplot(d1[feat], label=f'{label_v[1]}', bw=1.5, ax=ax1)
        sns.kdeplot(d0[feat], label=f'{label_v[0]}', bw=1.5, ax=ax1)
        plt.xlabel(feat, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=10, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=10)
    plt.show()


def plot_category_countplot(df, feats, label="flag", sub_col=2, figsize=(12,8)):
    '''
    绘制类别型特征分布
    :param df: DataFrame
    :param feats: list or ndarray
    :param label: string, the label of df
    :param sub_col: int
    :param figsize: tuple
    :return:
    '''
    if not isinstance(df, pd.DataFrame):
        raise Exception("The type of df must be DataFrame")
    sns.set(style='darkgrid', color_codes=True)
    plt.figure()
    plt.subplots(figsize=figsize)

    for i,feat in enumerate(feats):
        plt.subplot(sub_col, int(np.ceil(len(feats)/sub_col)), i+1)
        sns.countplot(x=feat, hue=label, data=df, palette='Set3')
    plt.show()


def plot_corr(df, feats, figsize=(12,12), mask=False):
    '''
    绘制相关性图
    :param df: pd.DataFrame
    :param feats: list or array
    :param figsize: tuple
    :param mask: bool
    :return:
    '''
    if not isinstance(df, pd.DataFrame):
        raise Exception("The type of df must be DataFrame")
    corr_df = df.loc[:,feats].corr()
    colormap = plt.cm.viridis
    plt.figure(figsize=figsize)
    plt.title('Heatmap', y=1.05, size=15)
    if mask:
        mask = np.triu(np.ones_like(corr_df,dtype=np.bool))
    sns.heatmap(corr_df, linewidths=0.1, vmax=1.0, mask=mask,
                square=True, cmap=colormap, linecolor='white', annot=True)
    plt.show()

def plot_bin_woe(binx, title=None, display_iv=False, figsize=(8, 6)):
    '''
    绘制woe
    :param binx: pd.DataFrame, the details of bins
    :param title: str, the title of the figure
    :param display_iv: bool, whether to show iv
    :param figsize: tuple, the size of figsize
    :return: figure
    '''
    y_right_max = np.ceil(binx['pct_1_row'].max() * 10)
    if y_right_max % 2 == 1:
        y_right_max = y_right_max + 1
    if y_right_max - binx['pct_1_row'].max() * 10 <= 0.3:
        y_right_max = y_right_max+2
    y_right_max = y_right_max/10
    if y_right_max > 1 or y_right_max <= 0 or y_right_max is np.nan or y_right_max is None:
        y_right_max = 1

    y_left_max = np.ceil(binx['pct_bin'].max() * 10) / 10
    if y_left_max > 1 or y_left_max <= 0 or y_left_max is np.nan or y_left_max is None:
        y_left_max = 1

    title_string = binx.loc[0, 'variable'] + "  (iv:" + str(binx.loc[0, 'IV']) + ")" \
                    if display_iv else binx.loc[0, 'variable']
    title_string = title + '-' + title_string if title is not None else title_string
    binx['good_distr'] = binx['num_0'] / np.sum(binx['num_01'])
    binx['bad_distr'] = binx['num_1'] / np.sum(binx['num_01'])
    ind = np.arange(len(binx.index))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    # ax1
    p1 = ax1.bar(ind, binx['good_distr'], width, color=(24 / 254, 192 / 254, 196 / 254))
    p2 = ax1.bar(ind, binx['bad_distr'], width, bottom=binx['good_distr'], color=(246 / 254, 115 / 254, 109 / 254))
    for i in ind:
        ax1.text(i, binx.loc[i, 'pct_bin'] * 1.02,
                 str(round(binx.loc[i, 'pct_bin'] * 100, 1)) + '%, ' + str(binx.loc[i, 'num_01']), ha='center')
    # ax2
    ax2.plot(ind, binx['pct_1_row'], marker='o', color='blue')
    for i in ind:
        ax2.text(i, binx.loc[i, 'pct_1_row'] * 1.02, str(round(binx.loc[i, 'pct_1_row'] * 100, 1)) + '%', color='blue',
                 ha='center')
    # settings
    ax1.set_ylabel('Bin count distribution', fontdict={"fontsize":10})
    ax2.set_ylabel('Bad probability', fontdict={"fontsize":10, "color":"blue"})
    ax1.set_yticks(np.arange(0, y_left_max + 0.2, 0.2))
    ax2.set_yticks(np.arange(0, y_right_max + 0.2, 0.2))
    ax2.tick_params(axis='y', colors='blue')
    plt.xticks(ind, binx.index)
    for xtick in ax1.get_xticklabels():
        xtick.set_rotation(30)

    plt.title(title_string, loc='left')
    plt.legend((p2[0], p1[0]), ('bad', 'good'), loc='upper right')
    plt.show()
    return fig
