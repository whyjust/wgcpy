#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Project:       wgcpy
@File Name:     main.py
@Author:        weiguang
@Date:          2021/6/29
"""
from sklearn.model_selection import train_test_split
from wgcpy.config import CONFIG
from wgcpy.preprocessing.eda import *
from wgcpy.preprocessing.data_dectection import DectectDF
from wgcpy.featureSelector.selector import *
from wgcpy.featureSelector.cal_iv_psi import *
from wgcpy.model.gen_pmml_model import *
from wgcpy.utils.ext_fn import *

pd.options.display.max_columns = 20
pd.options.display.max_rows = 20

def run(credit_data, numeric_feats, category_feats):
    # 数据EDA
    plot_feature_boxplot(credit_data, numeric_feats, save=True)
    plot_feature_distribution(credit_data, numeric_feats,
                              label="flag", sub_col=3, save=True)
    plot_category_countplot(credit_data, category_feats, label="flag",
                            sub_col=5, save=True, figsize=(20,12))
    plot_corr(credit_data, numeric_feats+['flag'], mask=True, save=True)

    # 数据分布
    with timer('detect dataframe'):
        dec = DectectDF(credit_data)
        df_des = dec.detect(special_value_dict={-999:np.nan},
                            output=r"./result")

    # 计算IV
    with timer("cal iv"):
        iv_details = cal_total_var_iv(credit_data,
                                      numeric_feats=numeric_feats,
                                      category_feats=category_feats,
                                      target='flag',
                                      max_interval=10,
                                      method='tree')
        fig = plot_bin_woe(binx=iv_details[iv_details['variable'] == 'credit.amount'],
                           title=None,
                           display_iv=True,
                           save=True)
        iv_details.to_csv(r'./result/iv_details.csv', index=False)

    # 计算psi
    with timer('cal psi'):
        except_array = credit_data['credit.amount'][:500]
        accept_arry = credit_data['credit.amount'][500:]
        psi_df = numeric_var_cal_psi(except_array,
                                     accept_arry,
                                     bins=10,
                                     bucket_type='bins',
                                     detail=True)
        psi_df.to_csv(r'./result/psi.csv')

    # 特征筛选&细筛
    with timer("cal cv score"):
        groups = credit_data['status.of.existing.checking.account']
        config = {
            "na_threshold": 0.95,
            "correlation_threshold": 0.6,
            "importance_cumsum_threshold": 0.95,
            "params": {
                "n_estimators": 200,
                "max_depth": 2,
                "learning_rate": 0.1,
                "boosting_type": "gbdt",
                "importance_type": "gain",
                "n_jobs": -1
            },
            "kfold": "StratifiedKFold",
            "groups": None,
            "categorical_feature": category_feats,
            "n_splits": 5,
            "incre_params": None,
            "total_iter": 20,
            "step": 1,
            "auc_interval": None
        }
        fs = FeatureSelector(data=credit_data,
                             target='flag',
                             base_features=numeric_feats+category_feats)
        fs.identify_all(config=config)
        fs.plot_feature_importance()
        fs.result_save(output=r"./result/feats_seletor_result.xlsx")

    # PMML建模与评估
    with timer("PMML model build"):
        trn_x, tes_x, y_trn, y_tes = train_test_split(credit_data,
                                                      credit_data['flag'],
                                                      test_size=0.2)
        pmml_model = genPMMLModel(data=trn_x,
                                  target="flag",
                                  base_features=numeric_feats+category_feats)

        pmml_model.make_pipeline_model(numeric_feature=numeric_feats,
                                       category_feature=category_feats,
                                       model_type='lgb',
                                       param_dict=config['params'])

        predict = pmml_model.evaluate(data=tes_x,
                                      target="flag")
        pmml_model.persist(base_dir="result",
                           model_name="credit")


if __name__ == "__main__":
    # 数据读取
    credit_data = pd.read_csv(r'data/germancredit.csv')
    numeric_feats = credit_data.select_dtypes(include=['int64','float64']).columns.tolist()
    category_feats = list(set(credit_data.columns) - set(numeric_feats + ['creditability']))
    credit_data["flag"] = credit_data["creditability"].replace({"good": 0, "bad": 1})

    # 运行
    run(credit_data=credit_data,
        category_feats=category_feats,
        numeric_feats=numeric_feats)
