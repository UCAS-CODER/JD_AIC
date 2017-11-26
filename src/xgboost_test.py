# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     xgboost_test
   Description :   尝试使用xgboost搭建框架
   Author :        wrm
   date：          2017/11/22
"""
__author__ = 'wrm'

import xgboost as xgb
import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import os
mingw_path = 'D:\\Program Files\\mingw-w64\\x86_64-7.2.0-posix-seh-rt_v5-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

# 自定义误差函数wmae
def error(y_hat, y):
    return 'error', float(sum(abs(y_hat-y.get_label()))) / sum(y.get_label())

now = time.time()
#sales_sum_csv = pd.read_csv("../data/t_sales_sum.csv")
#ads_csv = pd.read_csv("./pre/t_ads.csv")
#comment_csv = pd.read_csv("./pre/t_comment.csv")
#order_csv = pd.read_csv("./pre/t_order.csv")
#product_csv = pd.read_csv("./pre/t_product.csv")

data_csv = pd.read_csv("../data/pre_progress.csv")

#  共11列，分别代表 shop_id,1,2,3,4,8,9,10,11,12 月销售额，2,3,4月销售额之和
data_np = np.matrix(data_csv.as_matrix())
shop_id = data_np[:,0]
shop_id = [int(shop_id[i]) for i in range(len(shop_id))]
train = data_np[:,[5,6,7,8,9,1]]
labels = data_np[:,10]
test = data_np[:,[8,9,1,2,3,4]]

# 划分验证集
X_train, X_valid, y_train, y_valid = train_test_split(train, labels, test_size=0.33, random_state=42)

xgtrain = xgb.DMatrix(X_train, y_train)
xgval = xgb.DMatrix(X_valid, y_valid)
xgtest = xgb.DMatrix(test)

watchlist = [(xgtrain, 'train'),(xgval, 'val')]

# 正式进入xgboost模型
params={
'booster':'gbtree',
'objective': 'reg:linear',
'gamma':0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
'max_depth':6, # 构建树的深度 [1:]
#'lambda':100,  # L2 正则项权重
'subsample':1, # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
'colsample_bytree':0.7, # 构建树树时的采样比率 (0:1]
#'min_child_weight':7, # 节点的最少特征数
'eta': 0.001, # 如同学习率
'seed':710,
#'nthread':4,# cpu 线程数,根据自己U的个数适当调整
}
plst = list(params.items())
num_rounds = 5000  # 拟迭代次数

# 训练模型
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100, feval=error)

savetime = time.strftime('%m%d_%H_%M',time.localtime(time.time()))
model.save_model('../model/xgb_'+ savetime +'.model')  # 用于存储训练出的模型

#  预测5,6,7月的销量和
preds = model.predict(xgtest,ntree_limit=model.best_iteration)
result = pd.DataFrame(np.zeros([len(shop_id),2]), columns=['shop_id', 'preds'])
result['shop_id'] = shop_id
result['preds'] =  preds
result.to_csv('../submission/submission_'+ savetime +'.csv', sep=',', index=None, header=None)

cost_time = time.time()-now
print("Finish!",'\n',"cost time:",cost_time,"(s)......")