#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Project:       wgcpy
@File Name:     selector.py
@Author:        weiguang
@Date:          2021/6/23
"""
from pandas import ExcelWriter
from wgcpy.utils.ext_fn import *
from wgcpy.config import CONFIG
import matplotlib.pyplot as plt
from wgcpy.model.gen_model import GenCVModel, IncreaseCVSelector

logger = init_logger()

class FeatureSelector:
    def __init__(self, data, target, base_features):
        self.na_threshold = None
        self.labels = data.loc[:, target]
        self.data = data.loc[:, base_features]

        if not isinstance(data, pd.DataFrame):
            raise Exception("data must be DataFrame!")

        self.one_hot_features = None
        self.correlation_threshold = None
        self.importance_threshold = None
        self.useful_feats = None
        self.drop_feats = None

        self.record_na = None
        self.record_single_value = None
        self.record_collinear = None
        self.record_feats_importance = None
        self.record_cv_result = None
        self.record_increase_result = None
        self.record_increase_feats = None
        self.corr_matrix = None
        self.ops = {}


    def identify_na(self, na_threshold=1):
        '''
        缺失率剔除
        :param na_threshold: float, the threshold of missing rate
        :return:
        '''
        self.na_threshold = na_threshold
        na_rate_series = self.data.isna().sum() / self.data.shape[0]
        self.record_na = pd.DataFrame(data={"feature": na_rate_series.index, "na_fraction":na_rate_series.values})
        self.record_na = self.record_na.sort_values("na_fraction", ascending=False)
        self.record_na['mask'] = self.record_na['na_fraction'] >= na_threshold
        self.ops['drop_na_col'] = self.record_na[self.record_na['mask']]
        logger.info('%d features with greater than %0.2f missing values.\n' % (len(self.ops['drop_na_col']), self.na_threshold))


    def identify_single_value(self):
        '''
        剔除单一值
        :return:
        '''
        unique_counts = self.data.nunique()
        self.record_single_value = pd.DataFrame(data={'feature':unique_counts.index, 'nunique':unique_counts.values})
        self.record_single_value = self.record_single_value.sort_values('nunique', ascending=True)
        self.record_single_value['mask'] = self.record_single_value['nunique'] == 1
        self.ops['drop_single_col'] = self.record_single_value[self.record_single_value['mask']]
        logger.info('%d features with a single unique value.\n' % len(self.ops['drop_single_col']))


    def identify_collinear(self, correlation_threshold=1):
        '''
        剔除相关性
        :param correlation_threshold: float, the correlation threshold
        :return:
        '''
        self.correlation_threshold = correlation_threshold
        self.corr_matrix = self.data.corr()
        upper = self.corr_matrix.where(np.triu(np.ones(self.corr_matrix.shape), k=1).astype(np.bool))
        drop_corr_col = [i for i in upper.columns if np.any(upper[i].abs() > correlation_threshold)]

        self.record_collinear = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])
        for col in drop_corr_col:
            corr_feats = list(upper.index[upper[col].abs() > correlation_threshold])
            corr_values = list(upper[col][upper[col].abs() > correlation_threshold])
            drop_feats = [col for _ in range(len(corr_feats))]
            tmp = pd.DataFrame.from_dict({'drop_feature': drop_feats,
                                          'corr_feature': corr_feats,
                                          'corr_value': corr_values})
            self.record_collinear = self.record_collinear.append(tmp, ignore_index=True)
        self.ops['collinear'] = self.record_collinear
        logger.info('%d features with a correlation magnitude greater than %0.2f.\n' % (len(self.ops['collinear']),
                                                                                        self.correlation_threshold))


    def identify_cv_importance(self, params, kfold, categorical_feature=None, groups=None, n_splits=5):
        '''
        通过cv计算importance
        :param params: dict, the params of estimator
        :param kfold: str, the cv of split data
        :param categorical_feature: list, the category of feature
        :param groups: ndarray or Series
        :param n_splits: int
        :return:
        '''
        gen_model = GenCVModel(train_data=self.data, target_data=self.labels)
        self.record_feats_importance, self.record_cv_result = gen_model.cross_validation(params=params, kfold=kfold,
                                                                                         categorical_feature=categorical_feature,
                                                                                         groups=groups, n_splits=n_splits)
        self.ops['drop_zero_importance'] = self.record_feats_importance[self.record_feats_importance['importance'] == 0]


    def identify_increase_cv_feats(self, total_iter, step, categorical_feature, auc_interval, incre_params=None):
        """
        自动筛选特征，方向为验证集auc提升方向
        :param total_iter: int, the max iterator of traing round
        :param step: int, the numbers of add features
        :param categorical_feature: list, the category of features
        :param auc_interval: int, the miniumn interval of auc increase
        :param incre_params: dict, the params of increase model
        :return:
        """
        increase_model = IncreaseCVSelector(train_data=self.data, target_data=self.labels)
        self.record_increase_result, self.record_increase_feats = increase_model.get_lgb_cv_score(self.record_feats_importance,
                                                                                                  total_iter=total_iter,
                                                                                                  step=step,
                                                                                                  incre_params=incre_params,
                                                                                                  categorical_feature=categorical_feature,
                                                                                                  auc_interval=auc_interval)


    def identify_all(self, config=None):
        '''
        筛选特征
        :param config:
        :return:
        '''
        if config is None:
            config = CONFIG
        self.importance_threshold = config.get("importance_cumsum_threshold", CONFIG["importance_cumsum_threshold"])
        self.identify_na(na_threshold=config.get('na_threshold', CONFIG['na_threshold']))
        self.identify_single_value()
        self.identify_collinear(correlation_threshold=config.get("correlation_threshold", CONFIG["correlation_threshold"]))

        self.drop_feats = list(self.ops['drop_na_col'].feature) + list(self.ops["drop_single_col"].feature)\
                    + list(self.ops["collinear"].drop_feature)
        self.useful_feats = self.data.columns[~self.data.columns.isin(self.drop_feats)]
        logger.info(f"detect the total feats, drop feats: {len(self.drop_feats)}")
        logger.info(f"detect the total feats, userful feats: {len(self.useful_feats)}")

        self.data = self.data.loc[:, self.useful_feats]
        self.identify_cv_importance(
            params=config.get("params", CONFIG['params']),
            kfold=config.get("kfold", CONFIG["kfold"]),
            groups=config.get("groups", CONFIG["groups"]),
            categorical_feature=config.get("categorical_feature", CONFIG["categorical_feature"]),
            n_splits=config.get("n_splits", CONFIG["n_splits"])
        )

        self.record_feats_importance = self.record_feats_importance[self.record_feats_importance['cumsum']
                                                                    < self.importance_threshold]

        logger.info(f"After filtering cumsum importance above {'{:.2%}'.format(self.importance_threshold)}, "
                    f"The number of seleted features is: {len(self.record_feats_importance)} !")

        self.identify_increase_cv_feats(
            total_iter=config.get('total_iter', CONFIG['total_iter']),
            step=config.get("step", CONFIG["step"]),
            categorical_feature=config.get("categorical_feature", CONFIG["categorical_feature"]),
            auc_interval=config.get("auc_interval", CONFIG["auc_interval"])
        )


    def plot_feature_importance(self, n=50, figsize=(20, 12)):
        '''
        绘制特征重要度
        :param n: int, the top-n features
        :param figsize: tuple, the figsize of plot
        :return:
        '''
        plt.figure(figsize=figsize)
        ax = plt.subplot()

        ax.barh(list(reversed(list(self.record_increase_feats.index[:n]))),
                self.record_increase_feats['importance_normalized'].head(n),
                align='center', edgecolor='r')
        ax.set_yticks(list(reversed(list(self.record_increase_feats.index[:n]))))
        ax.set_yticklabels(self.record_increase_feats['feature'].head(n))

        plt.xlabel('Normalized Importance')
        plt.title(f"Feature Importances: {':.2%'.format(sum(self.record_increase_feats['importance_normalized']))}")
        plt.show()


    def result_save(self, output=None):
        '''
        结果文件保存
        :param output: string, the path of save result
        :return:
        '''
        if not output.endswith("xlsx"):
            raise AssertionError("output must be a file and endswith xlsx!")

        with ExcelWriter(output, engine="xlsxwriter", mode="w") as writer:
            self.record_cv_result.to_excel(excel_writer=writer,
                                           sheet_name=u"模型CV-Importance评估结果",
                                           index=True)
            self.record_feats_importance.to_excel(excel_writer=writer,
                                                  sheet_name=u"模型CV-{}Importance".format('{:.2%}'.format(self.importance_threshold)),
                                                  index=True)
            self.record_increase_feats.to_excel(excel_writer=writer,
                                                sheet_name="特征IncreaseCV-Filter评估结果",
                                                index=True)
            self.record_increase_result.to_excel(excel_writer=writer,
                                                 sheet_name="模型IncreaseCV-STEP验证结果",
                                                 index=True)

