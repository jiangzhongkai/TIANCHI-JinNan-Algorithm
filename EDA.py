"""-*- coding: utf-8 -*-
 DateTime   : 2019/1/2 14:18
 Author  : Peter_Bonnie
 FileName    : EDA.py
 Software: PyCharm
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#根据特征来可视化数据集
class EDA(object):
    """数据探索分析类"""
    def __init__(self):
        """初始化函数"""

    @staticmethod
    def load_Data(train,test):
        train=pd.read_csv(train)
        test=pd.read_csv(test)

        return train,test

    def show_nan_data(self,train,test,num_train,num_test):
        """
        根据num的数量来展示数据的缺失情况
        """
        ###展示训练集的数据
        stats = []
        for col in train.columns:
            stats.append((col, train[col].nunique(), train[col].isnull().sum() * 100 / train.shape[0],
                            train[col].value_counts(normalize=True, dropna=False).values[0] * 100, train[col].dtype))

        stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                                    'Percentage of values in the biggest category', 'type'])
        print(stats_df.sort_values('Percentage of missing values', ascending=False)[:num_train])

        ###展示测试集的数据
        stats = []
        for col in test.columns:
            stats.append((col, test[col].nunique(), test[col].isnull().sum() * 100 / test.shape[0],
                            test[col].value_counts(normalize=True, dropna=False).values[0] * 100, train[col].dtype))

        stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                                    'Percentage of values in the biggest category', 'type'])
        print(stats_df.sort_values('Percentage of missing values', ascending=False)[:num_test])

    def plot_figure_shoulv(self,train,test):

        train,test=self.load_Data(train,test)
        target_col = "收率"
        plt.figure(figsize=(8, 6))
        plt.scatter(range(train.shape[0]), np.sort(train[target_col].values))
        plt.xlabel('index', fontsize=12)
        plt.ylabel('yield', fontsize=12)
        plt.show()

        plt.figure(figsize=(12, 8))
        sns.distplot(train[target_col].values, bins=50, kde=False, color="red")
        plt.title("Histogram of yield")
        plt.xlabel('yield', fontsize=12)
        plt.show()

if __name__=="__main__":
    train=pd.read_csv('Data/jinnan_round1_train_20181227.csv', encoding = 'gb18030')
    test = pd.read_csv('Data/jinnan_round1_testA_20181227.csv', encoding='gb18030')
    # plt.figure()
    # train = train[train['收率'] > 0.84]
    # x=[i for i in range(train.shape[0])]
    # plt.scatter(x,train['收率'].values)
    # plt.show()
    # print(train.info())
    print(train.info())

    print("=========object==============")
    for col in train.columns:
        if train[col].dtype=="object":
            print(col)

    print("=======float64============")
    for col in train.columns:
        if train[col].dtype=="float64":
            print(col)

    print("=======int64=============")
    for col in train.columns:
        if train[col].dtype=="int64":
            print(col)

    for col in train.columns:
        print(col,train[col].unique(),train[col].isnull().sum())

    plt.figure()
    plt.subplot(231)
    plt.title("yeild distribution info")
    plt.hist(train['收率'].values, facecolor='red')

    plt.subplot(232)
    plt.title("Train_B14")
    plt.hist(train["B14"].values)

    plt.subplot(233)
    plt.title("Test_B14")
    plt.hist(test["B14"].values)


    plt.tight_layout()
    plt.show()

    print(train["B12"].value_counts())
    print(test["B12"].value_counts())
















