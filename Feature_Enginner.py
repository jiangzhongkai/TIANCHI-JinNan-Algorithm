"""-*- coding: utf-8 -*-
 DateTime   : 2019/1/1 14:09
 Author  : Peter_Bonnie
 FileName    : Feature_Enginner.py
 Software: PyCharm
 Desc :特征工程
"""
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold, RepeatedKFold
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from Data_Processing import *


class Features_Enginner(object):
    """
    特征工程
    """

    def __init__(self):
        pass

    def find_outlier_with_mean_std(self, train, col):
        """利用服从某个分布来寻找离群点"""
        temp_outlier = {}
        temp_index=[]
        mean = train[col].mean()
        std = train[col].std()
        for i in range(len(train[col])):
            print(i,train[col].values[i])
            if train[col].values[i] > mean + 3 * std or train[col].values[i] < mean - 3 * std:
                temp_outlier[i] = train[col].values[i]
                temp_index.append(i)
        return temp_outlier,temp_index

    def find_outlier_with_cluster(self, train):
        """利用聚类的方式来寻找离群点"""
        temp_outlier = {}

        pass

    def judge_detect(self, data):
        pass

    def target_encoding(self, target):
        """采用target mean value 来给类别特征做编码"""
        pass


def modeling_cross_validation(params, X, y, nr_folds=5):
    oof_preds = np.zeros(X.shape[0])
    # Split data with kfold
    folds = KFold(n_splits=nr_folds, shuffle=False, random_state=4096)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print("fold n°{}".format(fold_ + 1))
        trn_data = lgb.Dataset(X[trn_idx], y[trn_idx])
        val_data = lgb.Dataset(X[val_idx], y[val_idx])

        num_round = 20000
        clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000,
                        early_stopping_rounds=100)
        oof_preds[val_idx] = clf.predict(X[val_idx], num_iteration=clf.best_iteration)

    score = mean_squared_error(oof_preds, y)
    # score = mean_squared_error(oof_preds, target)

    return score / 2


def featureSelect(train,columns,target):
    init_cols = columns
    params = {'num_leaves': 120,
              'min_data_in_leaf': 30,
              'objective': 'regression',
              'max_depth': -1,
              'learning_rate': 0.05,
              "min_child_samples": 30,
              "boosting": "gbdt",
              "feature_fraction": 0.9,
              "bagging_freq": 1,
              "bagging_fraction": 0.9,
              "bagging_seed": 11,
              "metric": 'mse',
              "lambda_l1": 0.02,
              "verbosity": -1}
    best_cols = init_cols.copy()
    best_score = modeling_cross_validation(params, train[init_cols].values, target.values, nr_folds=5)
    print("初始CV score: {:<8.8f}".format(best_score))
    save_remove_feat=[] #用于存储被删除的特征
    for f in init_cols:

        best_cols.remove(f)
        score = modeling_cross_validation(params, train[best_cols].values, target.values, nr_folds=5)
        diff = best_score - score
        print('-' * 10)
        if diff > 0.0000002:
            print("当前移除特征: {}, CV score: {:<8.8f}, 最佳cv score: {:<8.8f}, 有效果,删除！！".format(f, score, best_score))
            best_score = score
            save_remove_feat.append(f)
        else:
            print("当前移除特征: {}, CV score: {:<8.8f}, 最佳cv score: {:<8.8f}, 没效果,保留！！".format(f, score, best_score))
            best_cols.append(f)
    print('-' * 10)
    print("优化后CV score: {:<8.8f}".format(best_score))

    return best_cols,save_remove_feat


if __name__ == "__main__":
    outlier = Features_Enginner().find_outlier_with_mean_std('Data/jinnan_round1_train_20181227.csv', "收率")

    # train = pd.read_csv('Data/jinnan_round1_train_20181227.csv', encoding='gb18030')
    # target = train['收率']
    # train.pop('收率')
    # train.pop('样本id')
    # best_features = featureSelect(train, target)
    print(outlier)
