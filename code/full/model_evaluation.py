# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 18:35:24 2019
评价模型在测试集上的效果
@author: tqd95
"""

import numpy as np
import pandas as pd
import time 

# 提取测试集中的正样本(label=1)
anime_clean = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\anime_cleaned.csv')
test = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test.pkl')
cond = test['start'] > '2017-06-30'
test.loc[cond,'is_new_anime'] = 1
test.loc[-cond,'is_new_anime'] = 0
test_positive = test.loc[test.my_watched_episodes > 0,['user_id','anime_id','my_score','my_status','is_new_anime']]

# 读取各路召回结果
test_recall = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test_recall.pkl')



# 把训练好的模型应用在最终测试集上,返回一个经过过滤后的结果
# 过滤规则是：对每一个用户,召回得分最高的10部新番和20部旧番
def get_recommend_result():
    '''
    @model: 训练好的XGBoost排序模型
    @test_fn: 最终测试集, 可以用来预测打分的(经过了onehot处理的)
    '''
    # test_recall_mapping: 各类召回算法的综合结果,包含四列：user_id,anime_recall,recall_channel和is_new_anime
    test_recall_mapping = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test_recall_channel_mapping.pkl')
    y_pred = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\result\incremental_pred\pred_36.csv')
    #y_pred = model.predict_proba(test_fn)[:,1]
    # 模型在测试集上的预测得分情况(预测样本的label=1的概率)
    test_recall_result = pd.concat([test_recall_mapping, y_pred], axis = 1) 
    
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

filtered_rec_result = get_recommend_result()






# 计算精准率和召回率
def get_precision_recall(filtered_rec_result, test_positive):
    '''
    @filtered_rec_result: 模型经过排序、过滤后的结果
    @test_positive: 测试集里的正样本,即每个用户最终观看了的动漫, 有标记是旧番还是新番
    '''
    rt = 0
    t = 0
    r = 0
    cnt = 1
    for uid in set(test_positive.user_id):
        if cnt % 499 == 1:
            print('epoch:',cnt,'/',49932,'...')
        tu = set(test_positive.loc[test_positive.user_id == uid, 'anime_id'])
        ru = set(filtered_rec_result.loc[filtered_rec_result.user_id == uid, 'anime_recall'])
        rt += len(ru & tu)
        r += len(ru)
        t += len(tu)
        cnt += 1
    recall = rt/t       # 召回率 0.03697487836565543
    precision = rt/r    # 精准率 0.08963515171828458
    return precision, recall

start = time.time()
precision, recall = get_precision_recall(filtered_rec_result, test_positive)
end = time.time()
print('time used:',(end-start)/60, 'minutes')   # 3.87 minutes


# 计算命中率。定义为：如果某个用户的推荐列表里有动漫被看了，则命中。
def get_hit_rate(filtered_rec_result, test_positive):
    '''
    @filtered_rec_result: 模型经过排序、过滤后的结果
    @test_positive: 测试集里的正样本,即每个用户最终观看了的动漫, 有标记是旧番还是新番
    '''
    cnt = 0
    for uid in set(test_positive.user_id):
        tu = set(test_positive.loc[test_positive.user_id == uid, 'anime_id'])
        ru = set(filtered_rec_result.loc[filtered_rec_result.user_id == uid, 'anime_recall'])
        if tu & ru:
            cnt += 1
    hit_rate = cnt/len(set(test_positive.user_id))
    return hit_rate

start = time.time()
hit_rate = get_hit_rate(filtered_rec_result, test_positive)
end = time.time()
print('time used:',(end-start)/60, 'minutes')   # 3.93 minutes

# 计算覆盖率(最原始的,即最后推荐的物品数/总的物品数)
coverage = len(set(filtered_rec_result.anime_recall))/len(anime_clean)


# 计算item分布的覆盖率
def get_entropy(filtered_rec_result):
    # 计算最终推荐列表中所有推荐物品的分布的信息熵(对数是以2为底的)
    df = filtered_rec_result.groupby('anime_recall').size().rename('cnt').to_frame()
    df['prob'] = df['cnt']/sum(df['cnt'])
    entropy = -sum(df['prob'] * (df['prob'].apply(lambda x:np.log2(x))))
    return entropy

entropy = get_entropy(filtered_rec_result)

# 计算基于type的覆盖率
def get_type_entropy(filtered_rec_result, anime_clean):
    df = pd.merge(filtered_rec_result, anime_clean.loc[:,['anime_id','type']], left_on = 'anime_recall', right_on = 'anime_id', how = 'left')
    df = df.drop('anime_id',axis = 1)
    res= df.groupby('type').size().rename('cnt').to_frame()
    res['prob'] = res['cnt']/sum(res['cnt'])
    entropy = -sum(res['prob'] * (res['prob'].apply(lambda x:np.log2(x))))
    return entropy
type_entropy = get_type_entropy(filtered_rec_result, anime_clean)