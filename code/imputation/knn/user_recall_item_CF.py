# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:17:19 2019
利用item-CF对KNN填充后的数据进行用户召回
@author: tqd95
"""

import json
import pandas as pd
import time

# 新的train_non_zero(原版的基础上加上了target_zero, 暂时不考虑那些50%比例以下的动漫)
train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\imputation\knn\train_non_zero.pkl')

# 计算每个用户看过的所有动漫
def get_user_watched_item(train):
    ## 返回一个series, index是user_id, value是用户看过的所有动漫(包括低分)
    user_stats = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\user_stats.pkl')
    user_stats = user_stats.loc[:, ['user_id']]
    user_watched_item = train.groupby('user_id').agg(lambda x: set(x.anime_id))
    user_watched_item = user_watched_item.iloc[:, 0]
    user_watched_item = pd.DataFrame(user_watched_item)
    user_watched_item = user_watched_item.reset_index()
    user_watched_item['num'] = user_watched_item['anime_id'].apply(lambda x: len(x))
    return user_watched_item.iloc[:, :2].set_index('user_id')['anime_id']

user_watched_item = get_user_watched_item(train_non_zero)
user_watched_item = dict(user_watched_item)


# 测试集单个用户召回
def user_recall(user_id, user_item, user_watched_item, W, K1, K2):
    '''
    user_id: 测试集的用户id
    user_item: 通过训练集生成的用户-动漫倒排表，即每个用户在训练集中喜欢了哪些动漫
    user_watched_item:  一个dict,形如:{user_id:{},...}，表示每个用户看过哪些动漫(有打分记录的)
    anime_list: 动漫矩阵, 用index对应anime_id
    W: 动漫相似矩阵
    K1: 计算用户兴趣的邻域大小
    K2: 召回的动漫数, 最后召回的动漫结果是以index的形式存储。
    '''
    # 如果user_id在训练集里没看过一部动漫,或是看过但是没喜欢过一部动漫,这种用户是无法作推荐的。。
    if not user_id in set(user_watched_item.keys()):
        return []
    elif not user_id in user_item.user_id.unique():
        return []

    user_liked_animes = set(user_item.iloc[:, :2].set_index('user_id')['anime_id'][user_id])
    user_watched_animes = user_watched_item[user_id]

    rec_result = {}
    # a_id是某部用户喜欢动漫(当前动漫)的anime_id
    for a_id in user_liked_animes:
        # 找到和当前动漫相似度最高的K1本动漫
        cnt = 0
        for j, wj in sorted(W[str(a_id)].items(), key=lambda x: x[1], reverse=True):
            if int(j) in user_watched_animes:  # 但是要先剔除用户已经在训练集里看过了的动漫(这些动漫最后不会被推荐的)
                continue
            rec_result.setdefault(int(j), 0)
            rec_result[int(j)] += wj
            cnt += 1
            if cnt >= K1:
                break
        # if len(rec_result) >=K1:
        #        break
    top_K2_animes = [i for i, w in sorted(rec_result.items(), key=lambda x:x[1], reverse=True)[:K2]]
    return top_K2_animes

# 对测试集用户，生成召回表
def get_test_recall(test, user_item, user_watched_item, W, K1=10, K2=20):
    '''
    @test: 不加任何加工的测试集;
    @user_item: 过滤后的用户-动漫倒排表
    @user_watched_item: 用户看过的动漫表,用以控制过滤,使得最后的推荐结果没有被用户看过
    @W:动漫相似矩阵
    @K1: 计算用户兴趣的邻域大小
    @K2: 召回的动漫数, 最后召回的动漫结果是以anime_id的形式存储。
    返回的数据集有两列:
    @user_id: 用户id
    @anime_recall: 召回的动漫id
    '''
    testing = pd.DataFrame({'user_id': test.user_id.unique()})
    testing['anime_recall'] = testing['user_id'].apply(user_recall, user_item=user_item, user_watched_item=user_watched_item, W=W, K1=K1, K2=K2)
    return testing

test = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test.pkl')
user_item = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\imputation\knn\user_item.pkl')
with open(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\imputation\knn\similarity_maxtrix.json', 'r', encoding='UTF-8') as f:
    W = json.load(f)


start = time.time()
test_recall = get_test_recall(test=test, user_item=user_item,
                              user_watched_item=user_watched_item, W=W, K1=10, K2=20)
end = time.time()
print('time used:', (end-start)/60, 'minutes')  # 8.3 minutes


# 把召回的动画id拆分成多行(这样一个user_id就会占多行)
def explode(df, group_by_field, explode_field):
    '''
    @group_field: 分组键, string形式
    @explode_field: 根据df的explode_field列拆分成多行, string形式
    '''

    df1 = df[explode_field].apply(pd.Series).stack().rename(explode_field)
    df2 = df1.to_frame().reset_index(1, drop=True)
    res = df2.join(df[group_by_field]).reset_index(drop=True)
    return res.loc[:, [group_by_field, explode_field]]


test_recall = explode(df=test_recall, group_by_field='user_id', explode_field='anime_recall')
test_recall['anime_recall'] = test_recall['anime_recall'].astype(int)
test_recall['recall_channel'] = 'item-CF'  # 添加一列channel注明召回渠道是item-CF
test_recall.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\imputation\knn\test_recall.pkl')
