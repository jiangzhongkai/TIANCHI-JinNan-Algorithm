"""-*- coding: utf-8 -*-
 DateTime   : 2019/1/1 14:09
 Author  : Peter_Bonnie
 FileName    : Data_Processing.py
 Software: PyCharm
 Desc: 主要处理数据
"""
from __future__ import print_function
import pandas as pd
import re
import warnings
import math
warnings.filterwarnings('ignore')


def fillna_with_mean(data,column,value):
    """利用均值填充"""
    mean_val=cal_col_mean(data,column)
    for i in range(len(data[column])):
        if data[column].values[i]==value:
            data[column].values[i]=mean_val
    return data

def time_2_sec(t):
    try:
        hours,minutes,sec=t.split(":")
    except:
        if t=="1900/1/9 7:00":
            return math.floor((7*3600)/3600)
        elif t=="1900/1/1 2:30":
            return math.floor((2*3600+30*60)/3600)
        elif t=="700":
            return math.floor((7*3600)/3600)
        elif t==-1:
            return -1
        else:
            return 0
    try:
        sec=math.floor((int(hours)*3600+int(minutes)*60+int(sec))/3600)
    except:
        return math.floor((30*60)/3600)
        # return 0
    return sec

def get_last_time(t):

    #需要考虑的情况就是后面的数字比前面的数字要小的情况
    try:
        start_hour,start_min,end_hour,end_min=re.split("[:,-]",t)
    except:
        if t=="14::30-15:30":
            return 3600/3600
        elif t=="13；00-14:00":
            return 3600/3600
        elif t=="13::30-15:00":
            return 5400/3600
        elif t=="16:30-17;30":
            return 3600/3600
        elif t=="6:00-8；00":
            return 7200/3600
        elif t=="20;30-22；00":
            return 5400/3600
        elif t=="20:00-22;00":
             return 7200/3600
        elif t=="21:00-22;00":
            return 3600/3600
        elif t=='22"00-0:00':
            return 7200/3600
        elif t=="2:00-3;00":
            return 3600/3600
        elif t=="1:30-3;00":
            return 5400/3600
        elif t=="15:00-1600":
            return 3600/3600
        elif t=="11;30-13；00":
            return 5400/3600
        elif t==-1:
            return -1
        else:
            return 30*60/3600

    try:
        # if int(end_hour)<int(start_hour):
        #     sec=(int(end_hour)*3600+int(end_min)*60-int(start_hour)*3600-int(start_min)*60)/3600+24
        # else:
        sec=(int(end_hour)*3600+int(end_min)*60-int(start_hour)*3600-int(start_min)*60)/3600
    except:
        if t=='19:-20:05':
            return (60*60+300)/3600
        else:
            return (30*60)/3600
            # return 0
    return sec

#只处理没有时段特征的数据
def find_anomaly_data(column,data):

    temp_dict={}  #用于存储不符合这样的这样格式的数据

    for i in range(len(data[column])):
        if len(data[column].values[i].split(':'))!=3:
            temp_dict[i]=data[column].values[i]

    return column,temp_dict


def find_anomaly_data_2(column,data):
    """寻找时间段上具有异常的数据"""
    temp_dict={}
    for i in range(len(data[column])):
        start,end=data[column].values[i].split('-')
        if len(start.split(":"))+len(end.split(":"))!=4 and len(start.split(":"))+len(end.split(":"))!=0:
            temp_dict[i]=data[column].values[i]

    return column,temp_dict

def find_ano_data(column,data):

    temp_dict={}
    for i in range(len(data[column])):
        if data[column].values[i].find('::')!=-1:
            temp_dict[i]=data[column].values[i]
    return column,temp_dict

def adjust_time(column,data):
     """如果发现时间数据的时段前面大于后面，要么就去掉数据，要么就进行调整"""
     temp_dict={}
     for i in range(len(data[column])):

         """需要考虑凌晨的数字比较小，所以需要进行特殊处理"""

     return  column,temp_dict

def drop_columns(train,test,columns=None):
    """删除某些列"""
    for df in [train,test]:
        try:
           df.drop(columns, axis=1, inplace=True)
        except:
            raise ValueError("有些列并不存在")
    return train,test

def remove_columns_with_rate(train,ratio):
    """删除缺失比率达到ratio的列"""
    temp_cols=list(train.columns)
    for col in train.columns:
        rate = train[col].value_counts(normalize=True, dropna=False).values[0]
        if rate > ratio:
            temp_cols.remove(col)
    return temp_cols

def Min_MAX(data,col):
    max = data[col].max()
    min = data[col].min()
    for i in range(len(data[col])):
        data[col].values[i] = (data[col].values[i] - min) / (max - min)
    return data


def Normalization(data,columns,fun_name):
    """利用相关函数对数据进行规范化操作"""
    for col in columns:
        try:
            data[col]=data[col].apply(fun_name)
            print(data[col])
            Min_MAX(data,col)
            # max = data[col].max()
            # min = data[col].min()
            # for i in range(len(data[col])):
            #     data[col].values[i] = (data[col].values[i] - min) / (max - min)
        except:
            raise NotImplementedError("columns is None")
    return data

def fenxiang(train,test,target,cate_columns):

    train['target'] = target
    train['intTarget'] = pd.cut(train['target'], 5, labels=False)
    train = pd.get_dummies(train, columns=['intTarget'])
    li = ['intTarget_0.0', 'intTarget_1.0', 'intTarget_2.0', 'intTarget_3.0', 'intTarget_4.0']
    mean_columns = []
    for f1 in cate_columns:
        cate_rate = train[f1].value_counts(normalize=True, dropna=False).values[0]
        if cate_rate < 0.90:
            for f2 in li:
                col_name = 'B14_to_' + f1 + "_" + f2 + '_mean'
                mean_columns.append(col_name)
                order_label = train.groupby([f1])[f2].mean()
                train[col_name] = train['B14'].map(order_label)
                miss_rate = train[col_name].isnull().sum() * 100 / train[col_name].shape[0]
                if miss_rate > 0.0:
                    train = train.drop([col_name], axis=1)
                    mean_columns.remove(col_name)
                else:
                    test[col_name] = test['B14'].map(order_label)
    for f1 in cate_columns:
        cate_rate = train[f1].value_counts(normalize=True, dropna=False).values[0]
        if cate_rate < 0.90:
            for f2 in li:
                col_name = 'B12_to_' + f1 + "_" + f2 + '_mean'
                mean_columns.append(col_name)
                order_label = train.groupby([f1])[f2].mean()
                train[col_name] = train['B12'].map(order_label)
                miss_rate = train[col_name].isnull().sum() * 100 / train[col_name].shape[0]
                if miss_rate > 0.0:
                    train = train.drop([col_name], axis=1)
                    mean_columns.remove(col_name)
                else:
                    test[col_name] = test['B12'].map(order_label)
    # train.drop(li + ['target'], axis=1, inplace=True)
    return train,test,mean_columns



"""计算缺失率=="""
def cal_null_ratio(data,column):
    return column,data[column].isnull().sum()/len(data[column])

"""计算每个列特征的平均值"""
def cal_col_mean(data,column):

    if data[column].dtype=="float64":
        sum=0.0
    elif data[column].dtype=="int64":
        sum=0
    count=0

    for i in range(len(data[column])):
        if data[column].values[i]!=-1.0:
            sum += data[column].values[i]
            count+=1

    return sum/count


"""计算时间之间的间隔"""
def get_gap_between_time(t1,t2):

    """
    example:
        00:00:00
        00:00:00-02:00:00
    """
    global sec
    t1=str(t1)
    t2=str(t2)
    if t1.find('-')!=-1 and t2.find('-')!=-1:
        try:
            _,_,end_hour_1,end_min_1=re.split("[:,-]",t1)
            start_hour_2,start_min_2,_,_=re.split("[:,-]",t2)
            if start_hour_2<end_hour_1:
                sec=(int(start_hour_2)*3600+int(start_min_2)*60-int(end_hour_1)*3600-int(end_min_1)*60+24*3600)/3600
            else:
                sec=(int(start_hour_2)*3600+int(start_min_2)*60-int(end_hour_1)*3600-int(end_min_1)*60)/3600
            return sec
        except:
            return -1
    elif t1.find('-')!=-1 and t2.find('-')==-1:
        try:
            _,_,end_hour_1,end_min_1=re.split("[:,-]",t1)
            start_hour_2,start_min_2,_=re.split("[:]",t2)
            if start_hour_2<end_hour_1:
                sec = (int(start_hour_2) * 3600 + int(start_min_2) * 60 - int(end_hour_1) * 3600 - int(end_min_1) * 60+24*3600) / 3600
            else:
                sec = (int(start_hour_2) * 3600 + int(start_min_2) * 60 - int(end_hour_1) * 3600 - int(end_min_1) * 60) / 3600
            return sec
        except:
            return -1
    elif t1.find('-')==-1 and t2.find('-')!=-1:
        try:
            end_hour_1,end_min_1,_=re.split("[:]",t1)
            start_hour_2,start_min_2,_,_=re.split("[:,-]",t2)
            if start_hour_2<end_hour_1:
                sec = (int(start_hour_2) * 3600 + int(start_min_2) * 60 - int(end_hour_1) * 3600 - int(end_min_1) * 60 + 24 * 3600) / 3600
            else:
                sec = (int(start_hour_2) * 3600 + int(start_min_2) * 60 - int(end_hour_1) * 3600 - int(end_min_1) * 60) / 3600
            return sec
        except:
            return -1
    elif t1.find('-')==-1 and t2.find('-')==-1:
        try:
            start_hour_1, start_min_1, _ = re.split("[:]", t1)
            end_hour_1, end_min_1, _ = re.split("[:]", t2)
            if end_hour_1<start_hour_1:
                sec=(int(end_hour_1)*3600+int(end_min_1)*60-int(start_hour_1)*3600-int(start_min_1)*60+24*3600)/3600
            else:
                sec=(int(end_hour_1)*3600+int(end_min_1)*60-int(start_hour_1)*3600-int(start_min_1)*60)/3600
            return sec
        except:
            return -1
    else:
        return -1

def get_feat_diff(data, column="B12"):
    data["id"] = data["样本id"]
    data = data.sort_values(by="id")
    del data["样本id"]
    temp = []
    for i in range(0, len(data[column]) - 1, 1):
        temp.append(abs(data[column].values[i + 1] - data[column].values[i]))
    temp.append(0)
    data[column + "diff"] = temp
    return data[column + "diff"]

if __name__=="__main__":
    train = pd.read_csv('Data/jinnan_round1_train_20181227.csv', encoding='gb18030')
    test = pd.read_csv('Data/jinnan_round1_testA_20181227.csv', encoding='gb18030')

    temp=[]
    for i,j in zip(train["A9"].values,train["A11"].values):

        sec=get_gap_between_time(i,j)
        temp.append(sec)

    print(len(temp))









