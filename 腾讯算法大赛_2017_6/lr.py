#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 20:57:30 2017

@author: wrm
"""

from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import pandas as pd
import time 
import numpy as np

now = time.time()
traincsv = pd.read_csv("./pre/train.csv") # 注意自己数据路径
adcsv = pd.read_csv("./pre/ad.csv")
usercsv = pd.read_csv("./pre/user.csv")
positioncsv = pd.read_csv("./pre/position.csv")
categorycsv = pd.read_csv("./pre/app_categories.csv")
usercatecsv = pd.read_csv("./pre/new/usercate.csv")
usercatecsv = usercatecsv.drop(['Unnamed: 0'],axis=1)
    

dataset = pd.merge(traincsv, adcsv, how='inner', on='creativeID',sort=False,  
          suffixes=('_x', '_y'), copy=True, indicator=False)  
dataset = pd.merge(dataset, usercsv, how='inner', on='userID',sort=False,  
          suffixes=('_x', '_y'), copy=True, indicator=False)  
dataset = pd.merge(dataset, positioncsv, how='inner', on='positionID',sort=False,  
          suffixes=('_x', '_y'), copy=True, indicator=False) 
dataset = pd.merge(dataset, categorycsv, how='inner', on='appID',sort=False,  
          suffixes=('_x', '_y'), copy=True, indicator=False) 
dataset = pd.merge(dataset, usercatecsv, how='inner', on='userID',sort=False,  
          suffixes=('_x', '_y'), copy=True, indicator=False) 
dataset = dataset.drop(['conversionTime'],axis=1) 

train = dataset.iloc[::3,1:]
labels = dataset.iloc[::3,0].values


testcsv = pd.read_csv("./pre/test.csv") # 注意自己数据路径
tests = pd.merge(testcsv, adcsv, how='inner', on='creativeID',sort=False,  
          suffixes=('_x', '_y'), copy=True, indicator=False)  
tests = pd.merge(tests, usercsv, how='inner', on='userID',sort=False,  
          suffixes=('_x', '_y'), copy=True, indicator=False)  
tests = pd.merge(tests, positioncsv, how='inner', on='positionID',sort=False,  
          suffixes=('_x', '_y'), copy=True, indicator=False) 
tests = pd.merge(tests, categorycsv, how='inner', on='appID',sort=False,  
          suffixes=('_x', '_y'), copy=True, indicator=False) 
tests = pd.merge(tests, usercatecsv, how='inner', on='userID',sort=False,  
          suffixes=('_x', '_y'), copy=True, indicator=False) 
#%%
del(traincsv)
del(adcsv)
del(usercsv)
del(positioncsv)
del(categorycsv)
del(dataset)
del(testcsv)
del(usercatecsv)
#test_id = range(len(tests))
test = tests.iloc[:,2:]
tests = tests[['instanceID','label']]

train.iloc[:,0] = train.iloc[:,0].values%10000/100*60+train.iloc[:,0]%100 #clickTime中将点击时间在每天的分钟数作为特征
test.iloc[:,0] = test.iloc[:,0].values%10000/100*60+test.iloc[:,0]%100

#from collections import Counter
#cnt = Counter(train.iloc[:,3])
#indice3 = np.array([2579,3322,2150,4867,3688,675,3347,6086,3150,4250,4657,2426,7619,2831,4455,3789,1400,7149,2891,4292])
#for i in range(a.shape[0]):
#    if i not in indice3:
#        i = 0
        
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
cat_train = train.iloc[:,[4,5,10,12,13,14,15,19,20]]
cat_train_matrix = ohe.fit_transform(cat_train).toarray()
del(cat_train)

#from sklearn.feature_extraction import DictVectorizer
#train.iloc[:,4] = ','.join(str(i) for i in train.iloc[:,4]).split(',')
#train.iloc[:,5] = ','.join(str(i) for i in train.iloc[:,5]).split(',')
#train.iloc[:,10] = ','.join(str(i) for i in train.iloc[:,10]).split(',')
#train.iloc[:,12] = ','.join(str(i) for i in train.iloc[:,12]).split(',')
#train.iloc[:,13] = ','.join(str(i) for i in train.iloc[:,13]).split(',')
#train.iloc[:,14] = ','.join(str(i) for i in train.iloc[:,14]).split(',')
#train.iloc[:,15] = ','.join(str(i) for i in train.iloc[:,15]).split(',')
#train.iloc[:,19] = ','.join(str(i) for i in train.iloc[:,19]).split(',')
#train.iloc[:,20] = ','.join(str(i) for i in train.iloc[:,20]).split(',')
#cat_train = train.iloc[:,[4,5,10,12,13,14,15,19,20]]
#dict_vec = DictVectorizer(sparse=False)
#cat_train_matrix = dict_vec.fit_transform(cat_train.to_dict(orient='record'))
#del(cat_train)
train = np.hstack((train.iloc[:,[0,1,2,3,6,7,8,9,11,16,17,18,21,22,23,24,25,26,
                                 27,28,29,30,31,32,33,34]].values,cat_train_matrix))
del(cat_train_matrix)
#train[(train[:,1]==0),1] = np.nan

#test.iloc[:,4] = ','.join(str(i) for i in test.iloc[:,4]).split(',')
#test.iloc[:,5] = ','.join(str(i) for i in test.iloc[:,5]).split(',')
#test.iloc[:,10] = ','.join(str(i) for i in test.iloc[:,10]).split(',')
#test.iloc[:,12] = ','.join(str(i) for i in test.iloc[:,12]).split(',')
#test.iloc[:,13] = ','.join(str(i) for i in test.iloc[:,13]).split(',')
#test.iloc[:,14] = ','.join(str(i) for i in test.iloc[:,14]).split(',')
#test.iloc[:,15] = ','.join(str(i) for i in test.iloc[:,15]).split(',')
#test.iloc[:,19] = ','.join(str(i) for i in test.iloc[:,19]).split(',')
#test.iloc[:,20] = ','.join(str(i) for i in test.iloc[:,20]).split(',')
cat_test = test.iloc[:,[4,5,10,12,13,14,15,19,20]]
cat_test_matrix = ohe.transform(cat_test).toarray()
#cat_test_matrix = ohe.transform(cat_test).toarray()
del(cat_test)
test = np.hstack((test.iloc[:,[0,1,2,3,6,7,8,9,11,16,17,18,21,22,23,24,25,26,
                                 27,28,29,30,31,32,33,34]].values,cat_test_matrix))
del(cat_test_matrix)
#%%
grd_lm = LogisticRegression()
grd_lm.fit(train,labels)
pred = grd_lm.predict_proba(test)[:, 1]
