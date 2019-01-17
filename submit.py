"""-*- coding: utf-8 -*-
 DateTime   : 2019/1/2 21:22
 Author  : Peter_Bonnie
 FileName    : submit.py
 Software: PyCharm
"""
import pandas as pd
def  get_result(data1,data2):
    data1=pd.read_csv(data1)
    data2=pd.read_csv(data2)
    submit=pd.DataFrame()
    submit['id']=data1['id']
    submit['target']=(data1['target']*0.65+data2['target']*0.35).apply(lambda x:round(x,3))
    submit.to_csv("merge.csv",header=False,index=False)

def modify_result(data1):
    test=pd.read_csv('Data/jinnan_round1_testA_20181227.csv', encoding = 'gb18030')
    sub=pd.read_csv(data1,header=None)
    test_dict={}

    for value in [280,385,290,785]:
        test_dict[value]=test[test["B14"]==value]['样本id'].index

    for value in test_dict.keys():
         if value==280:
             sub.iloc[test_dict[value],1]=0.947
         elif value==385 or value==785:
             sub.iloc[test_dict[value],1]=0.879
         elif value==390:
             sub.iloc[test_dict[value],1]=0.89

    sub[1]=sub[1].apply(lambda x:round(x,3))
    sub.to_csv("modify.csv",index=False)


if __name__=="__main__":
    get_result("result/xgb9.99009061823e-05--1546836748.csv","result/lgb0.000114373956908--1546830802.csv")