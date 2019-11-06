# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:13:32 2019
0分填充策略--item-based neighborhood CF
@author: tqd95
"""

import pandas as pd
import numpy as np
import time
import json
from tqdm import tqdm

# 读取非零打分训练集, 以及在knn填补中得到的目标填充集targte_zero(观看集数占比>=0.5)
train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_non_zero.pkl')  # 包含1236的非零分以及4
target_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\imputation\target_zero.pkl')

item_avg_score = train_non_zero.groupby('anime_id')['my_score'].mean()
user_score = train_non_zero.groupby('user_id').apply(lambda df:dict(zip(df.anime_id,df.my_score)))


# 这里的物品间相似度就不能用之前的做法了。。
# 选择的是普通的cosine相似度, 以及带中心化的pearson correlation
# 生成带打分的用户物品倒排表
def get_user_item(train):
    user_item = train.loc[train.label == 1, :].groupby('user_id') \
                    .apply(lambda df:dict(zip(df.anime_id,df.my_score))) \
                    .to_frame().reset_index() \
                    .rename(columns = {0: 'anime_id'})
    user_item['num'] = user_item['anime_id'].apply(lambda x: len(x))
    return user_item

user_item = get_user_item(train_non_zero)


# 1. cosine相似度
def item_cosine_similarity(user_item, item_avg_score):
    '''
    @user_item：用户物品倒排表,每个user_id对应一个dict, 形如{anime_id:score, ...}
    @item_avg_score：每个item的平均得分, 如果某部动漫的平均打分为零, 则不计算它的相似度
    返回结果：
    @item_norm: 每个item的评分向量(长度等于用户数量)经过中心化后的模的平方, 最后计算相似度的时候的分母的组成部分
    @W: 物品间皮尔逊相关系数

    需要注意的一点是, 对于那些在train_non_zero里打分全是0分的动漫, 就不能计算他们和其他动漫的相似度, 需要剔除。
    '''
    item_norm = dict()
    W = dict()
    subset = item_avg_score[item_avg_score==0].index    # subset中的anime_id都是打分全为0的,遇到这些需要跳过处理
    start = time.time()
    for index, row in tqdm(user_item.iterrows()):
        anime_list = row['anime_id']    # anime_list是一个dict, 形如:{aid: score,...}, score是该用户对该动漫的打分
        for i, rui in anime_list.items():
            if i in subset: 
                continue
            item_norm.setdefault(i, 0)
            item_norm[i] += rui ** 2
            for j, ruj in anime_list.items():
                if j == i or j in subset: 
                    continue
                W.setdefault(i, {})
                W[i].setdefault(j, 0)
                W[i][j] +=  rui * ruj
    
    for i,wi in W.items():
        W[i] = {j: wij/np.sqrt(item_norm[i] * item_norm[j]) for j, wij in W[i].items()}
    end = time.time()
    print('time used:', (end-start)/60, 'minutes')
    return W

W_items = item_cosine_similarity(user_item, item_avg_score)

path = r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\imputation\neighborhood\item_cosine\item_cosine_similarity.json'
with open(path, 'w') as outfile:
    json.dump(W_items, outfile)
    

# target_zero中，那些在W_item中没有出现过的anime_id，一共有44部。
# 其中有2部是出现过，但是打分全是零; 另外42部压根没在train_non_zero里出现过, 没有评分记录
# 下面从target_zero里把有关这些anime_id的记录去掉
num_44 = set(target_zero.anime_id) - set(item_avg_score[item_avg_score!=0].index)
res = pd.DataFrame()
for aid in num_44:
    res = res.append(target_zero.loc[target_zero.anime_id == aid,:])
    
new_target_zero = target_zero.loc[target_zero.index.difference(res.index),:]


# 根据基于item的邻域方法进行预测
def item_based_imputation(row, item_avg_score, user_score, W, K=10):
    '''
    @row:待预测得分的DataFrame的行, 通过row来获取user_id以及anime_id
    @item_avg_score: 每部动漫在train_non_zero中的平均得分
    @user_score: 每个用户对每部动漫的打分
    @W: 动漫间相似矩阵
    @K: item邻域的K值
    '''
    pred = 0            # 修正项的分子
    denominator = 0     # 修正项的分母
    u_id, a_id = row['user_id'], row['anime_id']
    # 需要判断一个问题：就是target_zero里的user_id在训练集里是否出现过？
    if u_id not in user_score.index:
        return item_avg_score.get(a_id, 999)
    for j, wj in sorted(W[str(a_id)].items(), key=lambda x: x[1], reverse=True)[:K]:
        if not user_score[u_id].get(int(j)):
            continue
        else:
            ruj = user_score[u_id].get(int(j))
            pred += wj * (ruj - item_avg_score[int(j)])
            denominator += wj
    if denominator > 0:
        return item_avg_score[a_id] + pred/denominator
    else:       # 如果target_zero里的某部动漫, 在item_avg_score里没有出现, 就返回异常值999
        return item_avg_score.get(a_id, 999)   

  
path = r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\imputation\neighborhood\item_cosine\item_cosine_similarity.json'
with open(path, 'r', encoding='UTF-8') as f:
    W = json.load(f)

start = time.time()
pred_res = new_target_zero.apply(item_based_imputation, axis = 1, item_avg_score = item_avg_score,
                                 user_score = user_score, W = W, K = 10)
end = time.time()
print('time used:', (end-start)/60, 'minutes')

pred_res = pred_res.to_frame().rename(columns = {0: 'my_score'})
pred_res.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\imputation\neighborhood\item_cosine\pred_score.pkl')
pred_res = pred_res['my_score'].apply(lambda x:0 if x<0 else x)


# 把预测的得分合并到new_target_zero中, 并覆盖原来的my_score
# 根据原有的label生成规则, 生成新的label, 同时添加weight = my_watched_episodes / episodes
# 最后把new_target_zero合并到原来的train_non_zero中去, 成为新的训练集骨架！

new_target_zero.drop('my_score', axis = 1, inplace = True)
new_target_zero = pd.concat([new_target_zero, pred_res], axis = 1)
new_target_zero = new_target_zero[['user_id', 'anime_id', 'my_watched_episodes', 'episodes', 'my_score', 'my_status',
                                   'last_update_date', 'title', 'score', 'rank', 'start', 'end']]
new_target_zero.loc[new_target_zero.my_score >= 8, 'label'] = 1
new_target_zero.loc[new_target_zero.my_score < 8, 'label'] = 0
new_target_zero['confidence_weight'] = new_target_zero['my_watched_episodes'] / new_target_zero['episodes']
train_non_zero['confidence_weight'] = train_non_zero['my_watched_episodes'] / train_non_zero['episodes']
train_non_zero = pd.concat([train_non_zero, new_target_zero], axis = 0)
train_non_zero.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\imputation\neighborhood\item_cosine\train_non_zero.pkl')


# 生成带权的新的user_item用户-物品倒排表
def get_user_item(train):
    user_item = train.loc[train.label == 1, :].groupby('user_id') \
                    .apply(lambda df:dict(zip(df.anime_id,df.confidence_weight))) \
                    .to_frame().reset_index() \
                    .rename(columns = {0: 'anime_id'})
    user_item['num'] = user_item['anime_id'].apply(lambda x: len(x))
    return user_item

user_item = get_user_item(train_non_zero)
user_item.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\imputation\neighborhood\item_cosine\user_item.pkl')


