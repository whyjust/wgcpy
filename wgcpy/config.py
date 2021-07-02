#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Project:       wgcpy
@File Name:     config.py
@Author:        weiguang
@Date:          2021/6/24
"""

CONFIG = {
    "na_threshold": 0.95,
    "correlation_threshold": 0.95,
    "importance_cumsum_threshold": 0.95,
    "params": {
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.1,
        "boosting_type": "gbdt",
        "importance_type": "gain",
        "n_jobs": -1
    },
    "kfold": "StratifiedKFold",
    "groups": None,
    "categorical_feature": None,
    "n_splits": 5,
    "incre_params":{
            "max_depth": 3,
            "learning_rate": 0.1,
            "num_boost_round": 200,
            "metrics": "auc",
            "verbose_eval": 200,
            "verbose": -1,
            "early_stopping_rounds": 100,
            "seed": 10
    },
    "total_iter": 100,
    "step": 2,
    "auc_interval": 0.001
}