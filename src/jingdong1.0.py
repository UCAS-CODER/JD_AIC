# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:49:57 2017

@author: chen
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:22:31 2017

@author: chen
"""
import sys 
#reload(sys)
#sys.setdefaultencoding('utf-8')

import xgboost as xgb
#import pandas as pd
import time 
import numpy as np
import csv

#将t_order处理成 字典的格式  如下  {'593': {'2016-08-03': {'offer_amt': '0.0','offer_cnt': '0','ord_cnt': '1','pid': '1565979','rtn_amt': '0.0','rtn_cnt': '0','sale_amt': '37.17','user_cnt': '1'}, '2016-08-04': {'offer_amt': '0.0',
   
def zidian(path="../data/t_order.csv"):
    new_dict = {}
    with open(path, 'r')as csv_file:
        data = csv.DictReader(csv_file, delimiter=",")
#        i=0
        for row in data:
#            i=i+1
            item = new_dict.get(row['shop_id'], dict())
            item[row['ord_dt']] = {k: row[k] for k in ('sale_amt','offer_amt','offer_cnt','rtn_cnt','rtn_amt','ord_cnt','pid','user_cnt')}
            new_dict[row['shop_id']] = item
#            if i>1000:
#               break
    return new_dict
#得到每个商店每个月的销量（2016年8月至2017年4月）    
def test():
    W=dict()
   
    dictt=zidian()
    for shopid,values in dictt.items():
        for date,order in values.items():
#            print date
            date=date.split('-')
            W.setdefault(shopid,{})
#            print date[1]
            W[shopid].setdefault(int(date[1]),0)
            W[shopid][int(date[1])]+=int(order['ord_cnt'])*float(order['sale_amt'])
            

    return  W 
#如果没有销量，该月销量处理为0
def tianbu():
    a=[1,2,3,4,8,9,10,11,12]
    WW=test()
    for shopid,values in WW.items():
        for x in a:
            WW[shopid].setdefault(x,0)
    return WW
#选出2016年8月至2017年1月的数据
def shaixuan():
    b=[2,3,4] 
    WWW=tianbu()
    for shopid,values in WWW.items():
        for x in b:
            if x in values.keys():
                del values[x]
    return WWW
#排序
def sortshaixuan():
    WWWW=shaixuan()
    S= sorted(WWWW.items(), key=lambda d:d[0], reverse = False)
    return S  
#将数据处理为列表的形式，便于进行矩阵转换      
def huan():
    V=[]
    WWW=sortshaixuan()
    for shopid,values in WWW:
        temp = sorted(values.items(), key=lambda d:d[0], reverse = False)
        V.append([temp[i][1] for i in range(9)])
    return V

#处理标签

#求出2017年2月3月4月的销量之和，作为标签
def label():
    Q=dict()
    b=[2,3,4]
    WL=tianbu()
    for shopid,values in WL.items():
        for date,order in values.items():
            if date in b:
              Q.setdefault(shopid,0)
              Q[shopid]+=values[date]
    return  Q
#排序
def sortlabel():
    a=label()
    dicttt= sorted(a.iteritems(), key=lambda d:d[0], reverse = False)
    return dicttt
#刚开始想把列表处理成[[][]...]的形式，后来觉得不如换成矩阵然后转置，更好处理，所以这里只有一个列表[]
def labellabel():
    l=[]
    QQ=sortlabel()
    for key,value in QQ:
        l.append(value)
    return l

now = time.time()


#正式进入xgboost模型
params={
'booster':'gbtree',
# 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
'objective': 'reg:linear', 
'eval_metric': 'mae',
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

#Using 10000 rows for early stopping. 
#offset = 1000000  # 训练集中数据50000，划分35000用作训练，15000用作验证
num_rounds = 5000 # 迭代拟次数

# 转成xgboost格式的矩阵
train=huan()
labels=labellabel()
# 划分训练集与验证集 
a=np.matrix(train)
b=np.matrix(labels)
c=b.T
xgtrain = xgb.DMatrix(a[::2,:], label=c[::2])
xgval = xgb.DMatrix(a[1::2,:], label=c[1::2])

# return 训练和验证的错误率
watchlist = [(xgtrain, 'train'),(xgval, 'val')]


# training model 
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, xgtrain, num_rounds, watchlist,early_stopping_rounds=100)
# 保存模型的时间
savetime = time.strftime('%m%d_%H:%M',time.localtime(time.time()))
#model.save_model('./model/xgb_'+ savetime +'.model') # 用于存储训练出的模型
#preds = model.predict(xgtest,ntree_limit=model.best_iteration)

#  选取测试数据的列，做为提交的文件
#tests['label'] = preds
#submission = tests[['instanceID','label']]
#submission.sort_values(by='instanceID',ascending=True,inplace=True)

# 将预测结果写入文件，方式有很多，自己顺手能实现即可
#np.savetxt('./submission/submission_'+ savetime +'.csv',submission,header="instanceID,prob",fmt='%d,%f')


cost_time = time.time()-now
print("end ......",'\n',"cost time:",cost_time,"(s)......")
    
if __name__=='__main__':
    huan()
    