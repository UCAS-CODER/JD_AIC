#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:27:32 2017

@author: wrm
"""

import xgboost as xgb
import pandas as pd
import time 
import numpy as np
from sklearn.feature_extraction import DictVectorizer

usercsv = pd.read_csv("./pre/user.csv")
categorycsv = pd.read_csv("./pre/app_categories.csv")

installedappcsv = pd.read_csv("./pre/user_installedapps.csv")
cate_install = pd.merge(installedappcsv, categorycsv, how='inner', on='appID',sort=False,  
          suffixes=('_x', '_y'), copy=True, indicator=False)  
cate_install['appID'] = 1
del(installedappcsv)
from numpy import *
usercate = pd.DataFrame(zeros([usercsv.shape[0],15]),
                        columns=['userID',0,101,104,106,108,2,201,203,209,301,402,407,408,503])
usercate['userID'] = usercsv['userID']
usercate = usercate.values
cate_install = cate_install.values
del(usercsv)

now = time.time()
for i in cate_install:
    if i[2] == 101:
        usercate[i[0]-1,2] += 1
    elif i[2] == 104:
        usercate[i[0]-1,3] += 1
    elif i[2] == 106:
        usercate[i[0]-1,4] += 1
    elif i[2] == 108:
        usercate[i[0]-1,5] += 1
    elif i[2] == 2:
        usercate[i[0]-1,6] += 1
    elif i[2] == 201:
        usercate[i[0]-1,7] += 1
    elif i[2] == 203:
        usercate[i[0]-1,8] += 1
    elif i[2] == 209:
        usercate[i[0]-1,9] += 1
    elif i[2] == 301:
        usercate[i[0]-1,10] += 1
    elif i[2] == 402:
        usercate[i[0]-1,11] += 1
    elif i[2] == 407:
        usercate[i[0]-1,12] += 1
    elif i[2] == 408:
        usercate[i[0]-1,13] += 1
    elif i[2] == 503:
        usercate[i[0]-1,14] += 1
    else:
        usercate[i[0]-1,1] += 1
cost_time = time.time()-now
print "end ......",'\n',"cost time:",cost_time,"(s)......"
            
save = pd.DataFrame(usercate,columns=['userID',0,101,104,106,108,2,201,203,209,301,402,407,408,503])
save.to_csv('./pre/new/usercate.csv')
