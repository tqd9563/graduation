# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 14:40:51 2019

@author: tqd95
"""

import pandas as pd
import numpy as np 
import xgboost as xgb
import pickle

anime_clean = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\anime_cleaned.csv')

'''
用原生的xgboost接口训练
一共训练了两个模型，分别对应incre_model_1和incre_model_2
模型参数主要为：100棵树，不设置min_child_weight，对于学习率eta:
model_sampling_1：eta = 0.5
model_sampling_2：eta = 0.2
'''

xgb_params_init = {'objective':'binary:logistic',
                   'eval_metric': 'auc',
                   'eta':0.2,
                   'max_depth':5,
                   #'min_child_weight':100,
                   'gamma':1,
                   'subsample':0.8,
                   'colsample_bytree':0.8,
                   'n_jobs':-1,
                   'alpha':0,
                   'lambda':1,
                   'seed':1001
                  }


train_fn = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_fn.pkl')
test_fn = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test_fn.pkl')
xg_test= xgb.DMatrix(test_fn.values, nthread=-1)
y_train = train_fn['label'].values
x_train = train_fn.drop('label',axis=1).values
xg_train = xgb.DMatrix(x_train, label = y_train)
watchlist = [(xg_train,'train')]


# 100棵树, min_child_weight=100, eta=0.5
model = xgb.train(xgb_params_init, dtrain = xg_train, num_boost_round = 100, evals = watchlist,
                  early_stopping_rounds = 10, verbose_eval = True)

auc = 0.852158
model_path = r'C:\Users\tqd95\Desktop\graduation_thesis\result\model_sampling_1.pkl'
with open(model_path, 'wb') as fw:
    pickle.dump(model, fw)


def get_recommend_result(model, xgtest):
    '''
    @model: 训练好的XGBoost排序模型
    @test_fn: 最终测试集, 可以用来预测打分的(经过了onehot处理的)
    '''
    # test_recall_mapping: 各类召回算法的综合结果,包含四列：user_id,anime_recall,recall_channel和is_new_anime
    test_recall_mapping = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test_recall_channel_mapping.pkl')
    y_pred = model.predict(xgtest)
    # 模型在测试集上的预测得分情况(预测样本的label=1的概率)
    test_recall_result = pd.concat([test_recall_mapping, pd.DataFrame(y_pred)], axis = 1) \
                            .rename(columns = {0:'pred_score'})
    
    # 定义一个函数,针对同一个user召回的旧番还是新番,分别召回K1和K2部
    def get_rec_result(df, K1=20, K2=10):
        df['ranking'] = df.groupby('is_new_anime')['pred_score'].rank(ascending = False, method = 'first')
        res = df.loc[((df.is_new_anime==0) & (df.ranking<=K1)) | ((df.is_new_anime==1) & (df.ranking<=K2)),:] \
            .sort_values(by = ['is_new_anime','ranking'], ascending=[True,True])
        return res

    res = test_recall_result.groupby('user_id').apply(get_rec_result, K1=20, K2=10)
    res = res.drop('user_id',axis=1) \
            .reset_index() \
            .drop('level_1',axis=1)
    res['recall_channel'] = res['recall_channel'].apply(lambda x:x[0])
    res['anime_recall'] = res['anime_recall'].astype(int)
    res['is_new_anime'] = res['is_new_anime'].astype(int)
    res['ranking'] = res['ranking'].astype(int)
    return res



filtered_rec_result = get_recommend_result(model, xg_test)