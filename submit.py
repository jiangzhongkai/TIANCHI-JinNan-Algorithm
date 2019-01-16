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


if __name__=="__main__":
    get_result("result/xgb9.99009061823e-05--1546836748.csv","result/lgb0.000114373956908--1546830802.csv")