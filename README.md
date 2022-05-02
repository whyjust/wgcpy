# wgcpy ![包名](https://raw.githubusercontent.com/whyjust/wgcpy/5698282f1959d02eb1ea5165c05cc910bc61f369/wgcpy/pic/python-wgcpy-green.svg)
> Data analysis and PMML model framework

> version 1.0.0

Different modules of the Package are provided for everyone to use.
- data detect 
- variable eda
- the way of cut variable bins
- calculate iv or psi
- auto feature selector
- generate PMML model

### Require
- python 3.5 or newer
- Java 1.8 or newer. The java executable must be available on system path.

### Install 
GitLab安装
```bash
pip install --upgrade https://github.com/whyjust/wgcpy
```

### Structure
wgcpy package tree structure:
```text
WGCPY
D:\GITHUB\WGCPY
│  .gitignore
│  info.log
│  LICENSE
│  main.py
│  MANIFEST.in
│  README.md
│  requirements.txt
│  setup.py
│
├─data
│
├─pic
│
├─result
│
└─wgcpy
    │  config.py
    │  __init__.py
    │
    ├─bins
    │      chi_merge.py
    │      cut_bins.py
    │      __init__.py
    │
    ├─featureSelector
    │      cal_iv_psi.py
    │      cal_iv_psi_special.py
    │      selector.py
    │      __init__.py
    │
    ├─model
    │      dz_eval.py
    │      gen_model.py
    │      gen_pmml_model.py
    │      __init__.py
    │
    ├─preprocessing
    │      baggingPU.py
    │      data_detection.py
    │      eda.py
    │      __init__.py
    │
    └─utils
            ext_fn.py
            __init__.py
```
### Usage
##### 1 main.py运行
```bash
python main.py
```
##### 2 数据EDA模块
```python
plot_feature_boxplot(credit_data, numeric_feats)
plot_feature_distribution(credit_data, numeric_feats,
                          label="flag", sub_col=3)
plot_category_countplot(credit_data, category_feats, label="flag",
                        sub_col=5, figsize=(20,12))
plot_corr(credit_data, numeric_feats+['flag'], mask=True)
```
##### 3 数据探查
```python
# 数据分布
with timer('detect dataframe'):
    dec = DetectDF(credit_data)
    df_des = dec.detect(special_value_dict={-999:np.nan},
                        output=os.path.join(base_dir, "result"))
```

##### Pubagging
```
with timer('pu bagging'):
    estimator = LGBMClassifier(n_estimators=200, max_depth=2, learning_rate=0.1)
    bc = BaggingClassifierPU(base_estimator=estimator, 
                            n_estimators = 30, 
                            n_jobs = -1, 
                            max_samples = len(credit_data[credit_data['flag']==1]))
    bc.fit(credit_data[numeric_feats], credit_data['flag'])
    score_arr = bc.oob_decision_function_[:,1]
    credit_data['score_pb'] = score_arr
    credit_data = credit_data[(credit_data['score_pb'].isna()) | (credit_data['score_pb']<0.9)]
    print('PUbagging-shape:', credit_data.shape)
```

##### 4 计算IV
```python
with timer("cal iv"):
    iv_details = cal_total_var_iv(credit_data,
                                  numeric_feats=numeric_feats,
                                  category_feats=category_feats,
                                  target='flag',
                                  max_interval=10,
                                  method='tree')
    fig = plot_bin_woe(binx=iv_details[iv_details['variable'] == 'credit.amount'],
                       title=None,
                       display_iv=True)
    iv_details.to_csv(os.path.join(base_dir,r'result\iv_details.csv'), index=False)
```
##### 5 计算PSI
```python
with timer('cal psi'):
    except_array = credit_data['credit.amount'][:500]
    accept_arry = credit_data['credit.amount'][500:]
    psi_df = numeric_var_cal_psi(except_array,
                                 accept_arry,
                                 bins=10,
                                 bucket_type='bins',
                                 detail=True)
    psi_df.to_csv(os.path.join(base_dir, r'result\psi.csv'))
```
##### 6 特征初筛与细筛
```python
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
    fs.result_save(output=os.path.join(base_dir, r".\result\feats_seletor_result.xlsx"))

```
##### 7 PMML建模与评估
```python
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
```

Let's started! Welcome to star!

