"""-*- coding: utf-8 -*-
 DateTime   : 2019/1/1 14:09
 Author  : Peter_Bonnie
 FileName    : Data_Processing.py
 Software: PyCharm
 Desc: 主要处理数据
"""
from __future__ import print_function
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

def time_2_sec(t):
    try:
        hours,minutes,sec=t.split(":")
    except:
        if t=="1900/1/9 7:00":
            return (7*3600)/3600
        elif t=="1900/1/1 2:30":
            return (2*3600+30*60)/3600
        elif t=="700":
            return (7*3600)/3600
        elif t==-1:
            return -1
        else:
            return 0
    try:
        sec=(int(hours)*3600+int(minutes)*60+int(sec))/3600
    except:
        return (30*60)/3600

    return sec

def get_last_time(t):

    try:
        start_hour,start_min,end_hour,end_min=re.split("[:,-]",t)
    except:
        if t=="14::30-15:30":
            return 3600/3600
        elif t=="13；00-14:00":
            return 3600/3600
        elif t=="13::30-15:00":
            return 5400/3600
        # elif t=="2:00-4::00":
        #     return 2
        # elif t=="18::30-20:00":
        #     return 5400/3600


        # elif t=="18:00:-18:30":
        #     return 30*60/3600
        # elif t=="15:30-17;30":
        #     return 2*3600/3600
        # elif t=="15；00-17:00":
        #     return 2*3600/3600
        # elif t=="20:00-22;00":
        #     return 2
        # elif t=="20；00-22:00":
        #     return 2
        # elif t=="12:00-14;00":
        #     return 2

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
        sec=(int(end_hour)*3600+int(end_min)*60-int(start_hour)*3600-int(start_min)*60)/3600
    except:
        if t=='19:-20:05':
            return (60*60+300)/3600
        else:
            return (30*60)/3600
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
         start_time,end_time=data[column].values[i].split('-')
         start_hour_time,start_min_time=start_time.split(":") if len(start_time.split(":"))==2 else 0,0
         end_hour_time,end_min_time=end_time.split(":") if len(end_time.split(":"))==2 else 0,0
         """需要考虑凌晨的数字比较小，所以需要进行特殊处理"""
         if int(start_hour_time)*3600+int(start_min_time)*60>int(end_hour_time)*3600+int(end_min_time)*60:
             # data[column].value[i]=end_time
             # data[column].value[i]+='-'
             # data[column].value[i]+=start_time
             temp_dict[i]=data[column].values[i]

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

def Normalization(data,columns,fun_name):
    """利用相关函数对数据进行规范化操作"""
    for col in columns:
        try:
            data[col]=data[col].apply(fun_name)
            max = data[col].max()
            min = data[col].min()
            for i in range(len(data[col])):
                data[col].values[i] = (data[col].values[i] - min) / (max - min)
        except:
            raise NotImplementedError("columns is None")
    return data

def modify_anomaly_data(t):
    pass

def punction_process(t):
    pass


if __name__=="__main__":
    data=pd.read_csv("Data/jinnan_round1_train_20181227.csv",encoding = 'gb18030')
    data['A24']=data['A24'].astype("str")
    print(data.dtypes)
    for i in ['A5','A9','A11','A14','A16','B7']:
        print(find_anomaly_data(i,data))
    data['B10']=data['B10'].astype(str)
    data['B11']=data['B11'].astype(str)
    for i in ['A20','A28','B4','B9']:
        print(find_anomaly_data_2(i,data))

    for i in ['A20','A28','B4','B9',"B11","B10"]:
        print(find_ano_data(i,data))






