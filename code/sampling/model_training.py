# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 14:50:25 2019
训练XGBoost模型
@author: tqd95
"""

import pandas as pd
import xgboost as xgb
import gc
import time
import pickle

# 读入训练数据集
train_fn = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\train_fn.pkl')
x_columns = [x for x in train_fn.columns if x!= 'label']
train_x = train_fn[x_columns]
train_y = train_fn['label']
del train_fn, x_columns
gc.collect()


# 先手动设置一些参数,看看训练时间要多久。。
model = xgb.XGBClassifier(objective = 'binary:logistic', learning_rate = 0.5, 
                          n_estimators = 50, max_depth = 5, min_child_weight = 100,
                          gamma = 1, subsample = 0.8, colsample_bytree = 0.8,
                          random_state = 1001, n_jobs = -1, reg_alpha = 0, reg_lambda = 1)

start = time.time()
model.fit(train_x,train_y)
end = time.time()
print('time used:',(end-start)/3600,'hours')  # 0.03383358511659834 hours


with open('xgboost.pkl','wb') as fw:
    pickle.dump(model,fw)



# 读入测试集
test_fn = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\test_fn.pkl')


with open('xgboost.pkl', 'rb') as fr:
    model = pickle.load(fr)

y_predict = model.predict(test_fn)
y_predict_proba = model.predict_proba(test_fn)
