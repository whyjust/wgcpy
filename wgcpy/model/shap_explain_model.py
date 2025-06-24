#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   shap_explain_model.py
@Time    :   2022/09/07 21:38:27
@Author  :   weiguang 
'''
import pandas as pd
import numpy as np
import shap

def pipeline_shap(pipeline, X_train, y_train, interaction=False, sample=None):
    '''
    解释onehot_pipeline对应shap值
    入参:
        pipeline: onehot_pipeline的返回对象
        X_train: 训练集的特征,pd.DataFrame格式
        Y_train: 训练集的目标
        interaction: 是否返回shap interaction values
        sample: 抽样数int或抽样比例float,不传入则不抽样
    出参:
        feature_values: 如传入sample则是抽样后的X_train,否则为X_train
        shap_values: pd.DataFrame格式shap values,如interaction传入True,则为shap interaction values
    '''    
    if isinstance(sample, int):
        feature_values = X_train.sample(n=sample)
    elif isinstance(sample, float):
        feature_values = X_train.sample(frac=sample)
    else:
        feature_values = X_train
        
    mapper = pipeline.steps[0][1]
    model = pipeline._final_estimator
    sort_cols, onehot_cols = [], []
    for i in mapper.features:
        sort_cols += i[0]
        if 'OneHot' in str(i[1]):
            onehot_cols += i[0]
    feature_values = feature_values[sort_cols]
    
    mapper.fit(X_train)
    X_train_mapper = mapper.transform(X_train)
    feature_values_mapper = mapper.transform(feature_values)
    model.fit(X_train_mapper, y_train)
    
    shap_values = pd.DataFrame(index=feature_values.index, columns=feature_values.columns)
    explainer = shap.TreeExplainer(model)
    if interaction:
        mapper_shap_values = explainer.shap_interaction_values(feature_values_mapper)
        col_index = 0
        for col in sort_cols:
            if col in onehot_cols:
                col_index_span = len(X_train[col].unique())
                shap_values[col] = mapper_shap_values[
                    :, col_index: col_index + col_index_span, col_index: col_index + col_index_span
                ].sum(2).sum(1)
                col_index += col_index_span
            else:
                shap_values[col] = mapper_shap_values[:, col_index, col_index]
                col_index += 1
    else:
        mapper_shap_values = explainer.shap_values(feature_values_mapper)
        if len(mapper_shap_values) == 2:
            mapper_shap_values = mapper_shap_values[1]
        col_index = 0
        for col in sort_cols:
            if col in onehot_cols:
                col_index_span = len(X_train[col].unique())
                shap_values[col] = mapper_shap_values[
                    :, col_index: col_index + col_index_span
                ].sum(1)
                col_index += col_index_span
            else:
                shap_values[col] = mapper_shap_values[:, col_index]
                col_index += 1
    return feature_values, shap_values

def shape_explain_model(pkl_model, trn_x, base_feature):
    explainer = shap.TreeExplainer(pkl_model.pipeline_model.named_steps["classifier"])
    shap_values = explainer.shap_values(pkl_model.pipeline_model.named_steps["mapper"].transform(trn_x[base_feature]))
    shap_train_df = pd.DataFrame([list(shap_values[n].values) for n in range(trn_x.shape[0])], columns=base_feature)

    from collections import Counter, defaultdict
    def sortedDictValues(adict, n):
        items = list(adict.items())
        items.sort(key=lambda x:x[1],reverse=True)
        return {key: value for key, value in items[:n]}

    shap_train_df['explain_top5'] = shap_train_df.apply(lambda x: sortedDictValues(np.abs(x).to_dict(), 5), axis=1)
    print(shap_train_df['explain_top5'])
    return shap_train_df
