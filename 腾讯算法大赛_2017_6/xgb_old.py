#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 00:26:29 2017

@author: wrm
"""

#%%
import xgboost as xgb
import pandas as pd
import time 
import numpy as np
from sklearn.feature_extraction import DictVectorizer

now = time.time()
traincsv = pd.read_csv("./pre/train.csv") # 注意自己数据路径
adcsv = pd.read_csv("./pre/ad.csv")
usercsv = pd.read_csv("./pre/user.csv")
positioncsv = pd.read_csv("./pre/position.csv")
categorycsv = pd.read_csv("./pre/app_categories.csv")

dataset = pd.merge(traincsv, adcsv, how='inner', on='creativeID',sort=False,  
          suffixes=('_x', '_y'), copy=True, indicator=False)  
dataset = pd.merge(dataset, usercsv, how='inner', on='userID',sort=False,  
          suffixes=('_x', '_y'), copy=True, indicator=False)  
dataset = pd.merge(dataset, positioncsv, how='inner', on='positionID',sort=False,  
          suffixes=('_x', '_y'), copy=True, indicator=False) 
dataset = pd.merge(dataset, categorycsv, how='inner', on='appID',sort=False,  
          suffixes=('_x', '_y'), copy=True, indicator=False) 
dataset = dataset.drop(['conversionTime'],axis=1) 

train = dataset.iloc[::2,1:].values
labels = dataset.iloc[::2,0].values


testcsv = pd.read_csv("./pre/test.csv") # 注意自己数据路径
tests = pd.merge(testcsv, adcsv, how='inner', on='creativeID',sort=False,  
          suffixes=('_x', '_y'), copy=True, indicator=False)  
tests = pd.merge(tests, usercsv, how='inner', on='userID',sort=False,  
          suffixes=('_x', '_y'), copy=True, indicator=False)  
tests = pd.merge(tests, positioncsv, how='inner', on='positionID',sort=False,  
          suffixes=('_x', '_y'), copy=True, indicator=False) 
tests = pd.merge(tests, categorycsv, how='inner', on='appID',sort=False,  
          suffixes=('_x', '_y'), copy=True, indicator=False) 

#test_id = range(len(tests))
test = tests.iloc[:,2:].values

#train.iloc[:,0] = train.iloc[:,0].values%10000/100*60+train.iloc[:,0]%100 #clickTime中将点击时间在每天的分钟数作为特征
#test.iloc[:,0] = test.iloc[:,0]%10000/100*60+test.iloc[:,0]%100


#%%
params={
'booster':'gbtree',
# 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
'objective': 'binary:logistic', 
'eval_metric': 'logloss',
'gamma':0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
'max_depth':6, # 构建树的深度 [1:]
#'lambda':450,  # L2 正则项权重
'subsample':0.7, # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
'colsample_bytree':0.7, # 构建树树时的采样比率 (0:1]
#'min_child_weight':12, # 节点的最少特征数
'eta': 0.1, # 如同学习率
#'seed':710,
#'nthread':4,# cpu 线程数,根据自己U的个数适当调整
}

plst = list(params.items())

#Using 10000 rows for early stopping. 
#offset = 1000000  # 训练集中数据50000，划分35000用作训练，15000用作验证
num_rounds = 500 # 迭代你次数
xgtest = xgb.DMatrix(test)

# 划分训练集与验证集 
xgtrain = xgb.DMatrix(train[::2,:], label=labels[::2])
xgval = xgb.DMatrix(train[1::2,:], label=labels[1::2])

# return 训练和验证的错误率
watchlist = [(xgtrain, 'train'),(xgval, 'val')]


# training model 
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, xgtrain, num_rounds, watchlist,early_stopping_rounds=20)
savetime = time.strftime('%m%d_%H:%M',time.localtime(time.time()))
model.save_model('./model/xgb_'+ savetime +'.model') # 用于存储训练出的模型
preds = model.predict(xgtest,ntree_limit=model.best_iteration)

tests['label'] = preds
submission = tests[['instanceID','label']]
submission.sort_values(by='instanceID',ascending=True,inplace=True)

# 将预测结果写入文件，方式有很多，自己顺手能实现即可
np.savetxt('./submission/submission_'+ savetime +'.csv',submission,header="instanceID,prob",fmt='%d,%f')


cost_time = time.time()-now
print "end ......",'\n',"cost time:",cost_time,"(s)......"