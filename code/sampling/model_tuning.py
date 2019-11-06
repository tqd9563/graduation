# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:47:22 2019
XGBoost调参
@author: tqd95
"""

import xgboost as xgb
import pandas as pd
import time 
from sklearn.model_selection import GridSearchCV
# 读入训练数据集
train_fn = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\train_fn.pkl')
x_columns = [x for x in train_fn.columns if x!= 'label']
train_x = train_fn[x_columns]
train_y = train_fn['label']
del train_fn, x_columns

# 这里填初始参数
model = xgb.XGBClassifier(objective = 'binary:logistic', learning_rate = 0.5, 
                          n_estimators = 200, max_depth = 5, min_child_weight = 100,
                          gamma = 1, subsample = 0.8, colsample_bytree = 0.8,
                          random_state = 1001, n_jobs = -1, reg_alpha = 0, reg_lambda = 1) 

def find_best_tree_nums(model, train_x, train_y, useTrainCV=True, cv_folds=5, early_stopping_rouns=50):
    if useTrainCV:
        xgb_params = model.get_xgb_params()
        xgtrain = xgb.DMatrix(data = train_x.values, label = train_y.values)  # 这里的data和label都是数组
        xgb_boost_rounds = model.get_xgb_params()['n_estimators']

    results = xgb.cv(params=xgb_params, dtrain=xgtrain, num_boost_round=xgb_boost_rounds, 
                     nfold=cv_folds, stratified=False, folds=None, metrics='auc', obj=None, 
                     feval=None, maximize=False, fpreproc=None, as_pandas=True, verbose_eval=None, 
                     seed=0, early_stopping_rounds=early_stopping_rouns, show_stdv=True, 
                     callbacks=None, shuffle=True)
    return results
cv_results = find_best_tree_nums(model, train_x, train_y)

# 把最优参数传回模型中
model.set_params(n_estimators = cv_results.shape[0])






# 还是试试网格搜索吧
# 先确定树的棵树

param_test1 = {'n_estimators':range(50,200,20)}
gsearch1 = GridSearchCV(estimator=xgb.XGBClassifier(objective = 'binary:logistic', learning_rate = 0.5, 
                                                  max_depth = 5, min_child_weight = 100, gamma = 1, 
                                                  subsample = 0.8, colsample_bytree = 0.8, random_state = 1001, 
                                                  n_jobs = -1, reg_alpha = 0, reg_lambda = 1),
                        param_grid=param_test1,
                        scoring='roc_auc',
                        iid=False,cv=5,n_jobs=-1)

start = time.time()
gsearch1.fit(train_x,train_y)
end = time.time()
print('time used:', (end-start)/60, 'min')


gsearch1.best_params_, gsearch1.best_score_











