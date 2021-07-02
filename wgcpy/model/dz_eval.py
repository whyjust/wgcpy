#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Project:       wgcpy
@File Name:     dz_eval.py
@Author:        weiguang
@Date:          2021/6/28
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc,precision_recall_curve,\
                            average_precision_score,roc_auc_score
from sklearn.model_selection import learning_curve,validation_curve
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    '''
    混淆矩阵绘制
    :param y_true: ndarray, the true label
    :param y_pred: ndarray, the predict label
    :param classes: list, the class of label
    :param normalize: bool
    :param title: str
    :param cmap:
    :return:
    '''
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


def plot_ks_curve(preds, labels, is_score=False, n=100,
                  return_value=True, return_graph=False, return_table=False):
    '''
    KS绘制
    :param preds: ndarray, the predict labels
    :param labels: ndarray, the true labels
    :param is_score: bool, whether the predict labels is socre
    :param n: int, the bins of split
    :param return_value: bool, whether need to return value
    :param return_graph: bool, whether need to return graph
    :param return_table: bool, whether need to return detail table
    :return:
    '''
    ksds = pd.DataFrame({'bad': labels, 'pred': preds})
    ksds['good'] = 1 - ksds.bad

    if is_score:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, True])
    else:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, True])
    ksds1.index = range(len(ksds1.pred))
    ksds1['cumsum_good1'] = 1.0 * ksds1.good.cumsum() / sum(ksds1.good)
    ksds1['cumsum_bad1'] = 1.0 * ksds1.bad.cumsum() / sum(ksds1.bad)

    if is_score:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, False])
    else:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, False])
    ksds2.index = range(len(ksds2.pred))
    ksds2['cumsum_good2'] = 1.0 * ksds2.good.cumsum() / sum(ksds2.good)
    ksds2['cumsum_bad2'] = 1.0 * ksds2.bad.cumsum() / sum(ksds2.bad)

    # ksds1 ksds2 -> average
    ksds = ksds1[['cumsum_good1', 'cumsum_bad1']]
    ksds['cumsum_good2'] = ksds2['cumsum_good2']
    ksds['cumsum_bad2'] = ksds2['cumsum_bad2']
    ksds['cumsum_good'] = (ksds['cumsum_good1'] + ksds['cumsum_good2']) / 2
    ksds['cumsum_bad'] = (ksds['cumsum_bad1'] + ksds['cumsum_bad2']) / 2

    # ks
    ksds['ks'] = ksds['cumsum_bad'] - ksds['cumsum_good']
    ksds['tile0'] = range(1, len(ksds.ks) + 1)
    ksds['tile'] = 1.0 * ksds['tile0'] / len(ksds['tile0'])

    qe = list(np.arange(0, 1, 1.0 / n))
    qe.append(1)
    qe = qe[1:]

    ks_index = pd.Series(ksds.index)
    ks_index = ks_index.quantile(q=qe)
    ks_index = np.ceil(ks_index).astype(int)
    ks_index = list(ks_index)

    ksds = ksds.loc[ks_index]
    ksds = ksds[['tile', 'cumsum_good', 'cumsum_bad', 'ks']]
    ksds0 = np.array([[0, 0, 0, 0]])
    ksds = np.concatenate([ksds0, ksds], axis=0)
    ksds = pd.DataFrame(ksds,
                        columns=['tile', 'cumsum_good', 'cumsum_bad', 'ks'])

    ks_value = ksds.ks.max()
    ks_pop = ksds.tile[ksds.ks.idxmax()]
    if return_graph:
        plt.figure(figsize=(6,4))
        print('ks_value is ' + str(np.round(ks_value, 4)) + ' at pop = ' + str(
        np.round(ks_pop, 4)))
        # chart
        plt.plot(ksds.tile, ksds.cumsum_good, label='cum_good',
                 color='blue', linestyle='-', linewidth=2)
        plt.plot(ksds.tile, ksds.cumsum_bad, label='cum_bad',
                 color='red', linestyle='-', linewidth=2)
        plt.plot(ksds.tile, ksds.ks, label='ks',
                 color='green', linestyle='-', linewidth=2)
        plt.axvline(ks_pop, color='gray', linestyle='--')
        plt.axhline(ks_value, color='green', linestyle='--')
        plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_good'], color='blue',
                    linestyle='--')
        plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_bad'], color='red',
                    linestyle='--')
        plt.title('KS=%s ' % np.round(ks_value, 4) +
                  'at Pop=%s' % np.round(ks_pop, 4), fontsize=15)
        plt.show()
    if return_value:
        return ks_value
    elif return_table:
        return ksds, ks_value
    else:
        assert (return_table or return_graph or return_value),'请设置参数为True'
        return None


def plot_roc_curve(y_true, y_predict, return_value=True, return_graph=False):
    '''
    ROC曲线
    :param y_true: ndarray, the true labels
    :param y_predict: ndarray, the predict labels
    :param return_value: bool, whether need to return value
    :param return_graph: bool, whether need to return graph
    :return:
    '''
    fpr, tpr, threshold = roc_curve(y_true, y_predict)
    roc_auc = auc(fpr, tpr)
    if return_graph:
        plt.figure()
        plt.figure(figsize=(6, 4))
        lw=2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Roc Curve')
        plt.legend(loc="lower right")
        plt.show()
    if return_value:
        return roc_auc


# 绘制验证曲线
def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=(0, 1), cv=5, scoring='auc'):
    '''
    用于绘制验证曲线
    :param estimator: model
    :param title: str
    :param X: pd.DataFrame
    :param y: pd.Series or ndarray
    :param param_name: str
    :param param_range: list
    :param ylim: tuple (0, 1)
    :param cv: int
    :param scoring: str, the type of evaluation
    :return: pd.DataFrame
    '''
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel(scoring)
    if ylim is not None:
        plt.ylim(*ylim)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
    return pd.DataFrame({'train_scores_mean': train_scores_mean, 'train_scores_std': train_scores_std,
                         'test_scores_mean': test_scores_mean, 'test_scores_std': test_scores_std})


def plot_learning_curve(estimator, title, X, y, ylim=(0,1), cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5), scoring='roc_auc'):
    '''
    绘制学习曲线
    :param estimator: model
    :param title: str
    :param X: pd.DataFrame
    :param y: pd.Series or ndarray
    :param ylim: tuple or list, default is (0,1)
    :param cv: int
    :param train_sizes: list
    :param scoring: the type of evaluation
    :return: pd.DataFrame
    '''
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=6, scoring=scoring, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return pd.DataFrame({'train_scores_mean': train_scores_mean, 'train_scores_std': train_scores_std,
                         'test_scores_mean': test_scores_mean, 'test_scores_std': test_scores_std})
