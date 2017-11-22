#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 18:46:02 2017

@author: wrm
"""
#%%
import xgboost as xgb
import pandas as pd
import time 
import numpy as np

now = time.time()
traincsv = pd.read_csv("./pre/train.csv") # 注意自己数据路径
adcsv = pd.read_csv("./pre/ad.csv")  # 广告
usercsv = pd.read_csv("./pre/user.csv")  # 用户
positioncsv = pd.read_csv("./pre/position.csv")  #  位置
categorycsv = pd.read_csv("./pre/app_categories.csv")  # 种类
usercatecsv = pd.read_csv("./pre/new/usercate.csv")
usercatecsv = usercatecsv.drop(['Unnamed: 0'],axis=1)
    
#  表的拼接
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

#  数据隔3取1，为了避免内存不够，训练数据丢掉第一列
train = dataset.iloc[::3,1:]
labels = dataset.iloc[::3,0].values


testcsv = pd.read_csv("./pre/test.csv") # 注意自己数据路径
#  测试数据拼接表格
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

#  处理时间特征
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
#  类别特征onehot处理
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

#  拼接数值特征和onehot特征
train = np.hstack((train.iloc[:,[0,1,2,3,6,7,8,9,11,16,17,18,21,22,23,24,25,26,
                                 27,28,29,30,31,32,33,34]].values,cat_train_matrix))
del(cat_train_matrix)
#from sklearn import preprocessing
#scaler = preprocessing.StandardScaler()
#train[:,:2] = scaler.fit_transform(train[:,:2])
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

#  测试数据类别特征onehot处理
cat_test = test.iloc[:,[4,5,10,12,13,14,15,19,20]]
cat_test_matrix = ohe.transform(cat_test).toarray()
#cat_test_matrix = ohe.transform(cat_test).toarray()
del(cat_test)

#  拼接特征
test = np.hstack((test.iloc[:,[0,1,2,3,6,7,8,9,11,16,17,18,21,22,23,24,25,26,
                                 27,28,29,30,31,32,33,34]].values,cat_test_matrix))
del(cat_test_matrix)
#test = scaler.transform(test)

#  选择这些列的特征，是根据之前训练结果的特征贡献度选取的
feat_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,32,33,34,
              37,38,39,40,41,42,43,44,45,48,49,50,51,52,53,59,69,70,
              72,75,76,77]
#train = train[:,feat_index]
#test = test[:,feat_index]


#train = train.drop(['userID'],axis=1) 
#test = test.drop(['userID'],axis=1) 
#train.iloc[:,[2,3,4,5,6,7,8,9,11,12,13,15,16,17,18]] = train.iloc[:,[2,3,4,5,6,7,8,9,11,12,13,15,16,17,18]].astype(object)
#test.iloc[:,[2,3,4,5,6,7,8,9,11,12,13,15,16,17,18]] = test.iloc[:,[2,3,4,5,6,7,8,9,11,12,13,15,16,17,1]].astype(object)

#train.iloc[:,3] = ','.join(str(i) for i in train.iloc[:,3]).split(',')#把int转为str才能变成类别特征
#train.iloc[:,4] = ','.join(str(i) for i in train.iloc[:,4]).split(',')
#train.iloc[:,11] = ','.join(str(i) for i in train.iloc[:,11]).split(',')
#dict_vec = DictVectorizer()
#train=dict_vec.fit_transform(train.to_dict(orient='record'))
#dict_vec.feature_names_
#%%
params={
'booster':'gbtree',
# 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
'objective': 'binary:logistic', 
'eval_metric': 'logloss',
'gamma':0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
'max_depth':6, # 构建树的深度 [1:]
#'lambda':100,  # L2 正则项权重
'subsample':1, # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
'colsample_bytree':0.7, # 构建树树时的采样比率 (0:1]
#'min_child_weight':7, # 节点的最少特征数
'eta': 0.1, # 如同学习率
'seed':710,
#'nthread':4,# cpu 线程数,根据自己U的个数适当调整
}

plst = list(params.items())

#Using 10000 rows for early stopping. 
#offset = 1000000  # 训练集中数据50000，划分35000用作训练，15000用作验证
num_rounds = 5000 # 迭代拟次数

#  pandas格式转成xgboost格式的矩阵
xgtest = xgb.DMatrix(test)

# 划分训练集与验证集 
xgtrain = xgb.DMatrix(train[::2,:], label=labels[::2])
xgval = xgb.DMatrix(train[1::2,:], label=labels[1::2])

# return 训练和验证的错误率
watchlist = [(xgtrain, 'train'),(xgval, 'val')]


# training model 
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, xgtrain, num_rounds, watchlist,early_stopping_rounds=20)
# 保存模型的时间
savetime = time.strftime('%m%d_%H:%M',time.localtime(time.time()))
model.save_model('./model/xgb_'+ savetime +'.model') # 用于存储训练出的模型
preds = model.predict(xgtest,ntree_limit=model.best_iteration)

#  选取测试数据的列，做为提交的文件
tests['label'] = preds
submission = tests[['instanceID','label']]
submission.sort_values(by='instanceID',ascending=True,inplace=True)

# 将预测结果写入文件，方式有很多，自己顺手能实现即可
np.savetxt('./submission/submission_'+ savetime +'.csv',submission,header="instanceID,prob",fmt='%d,%f')


cost_time = time.time()-now
print "end ......",'\n',"cost time:",cost_time,"(s)......"