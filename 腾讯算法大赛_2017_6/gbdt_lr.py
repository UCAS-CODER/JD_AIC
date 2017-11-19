#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 20:19:53 2017

@author: wrm
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve
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

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
train = min_max_scaler.fir_transform(train)
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
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
train = min_max_scaler.fit_transform(train)
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
test = min_max_scaler.transform(test)
    
feat_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,32,33,34,35,
              37,38,39,40,41,42,43,44,45,48,49,50,51,52,53,54,55,56,59,60,61,62,63,64,65,66,67,68,69,70,
              71,72,74,75,76,77,78]
train = train[:,feat_index]
test = test[:,feat_index]
#%%
# 弱分类器的数目
n_estimator = 10
# 随机生成分类数据。
X, y = make_classification(n_samples=8000000)  
# 切分为测试集和训练集，比例0.5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
# 将训练集切分为两部分，一部分用于训练GBDT模型，另一部分输入到训练好的GBDT模型生成GBDT特征，然后作为LR的特征。这样分成两部分是为了防止过拟合。
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)
# 调用GBDT分类模型。
grd = GradientBoostingClassifier(n_estimators=n_estimator)
# 调用one-hot编码。
grd_enc = OneHotEncoder()
# 调用LR分类模型。
grd_lm = LogisticRegression()


'''使用X_train训练GBDT模型，后面用此模型构造特征'''
grd.fit(X_train, y_train)

# fit one-hot编码器
grd_enc.fit(grd.apply(X_train)[:, :, 0])

''' 
使用训练好的GBDT模型构建特征，然后将特征经过one-hot编码作为新的特征输入到LR模型训练。
'''
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
# 用训练好的LR模型多X_test做预测
y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
# 根据预测结果输出
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)