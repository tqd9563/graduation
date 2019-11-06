# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:52:23 2019
利用item cosine距离给新、老用户推荐新番
@author: tqd95
"""

import json
import time
import pandas as pd


## 为老用户(即在训练集里有喜欢过的动漫的用户)推荐新番
## 平均每人推荐20部新番
def old_user_user_new_recall(user_id, user_item, W, K1=10, K2=20):
    # 如果用户没有喜欢过的动漫,那就不好推荐了。。
    if not user_id in user_item.user_id.unique(): 
        return []
    
    rec_result = {}
    user_liked_animes = user_item.iloc[:, :2].set_index('user_id')['anime_id'][user_id]

    # a_id是某部用户喜欢动漫(当前动漫)的anime_id
    for a_id in user_liked_animes:
        #index_i = str(mapping[a_id])   # 当前动画对应的index 
        # 寻找当前旧番和其他新番的相似系数, 注意下面的矩阵W如果是读取的本地json, 那key变成了string形式, 需要用str转换
        # 找到和当前动漫相似度最高的K1本动漫
        for j, wj in sorted(W[str(a_id)].items(), key=lambda x:x[1], reverse=True)[:K1]: 
            rec_result.setdefault(int(j), 0)
            rec_result[int(j)] += wj
            
    top_K2_new_anime = [i for i, w in sorted(rec_result.items(), key=lambda x:x[1], reverse=True)[:K2]]
    return top_K2_new_anime

# 对测试集用户，生成召回表
def get_test_old_user_new_recall(test, user_item, W, K1 = 10, K2 = 20):
    '''
    @test: 不加任何加工的测试集;
    @user_item: 过滤后的用户-动漫倒排表
    @W: 老番-新番间cosine距离矩阵
    @K1: 计算用户兴趣的邻域大小
    @K2: 召回的动漫数, 最后召回的动漫结果是以anime_id的形式存储。
    返回的数据集有两列:
    @user_id: 用户id
    @anime_recall: 召回的动漫id
    '''
    testing = pd.DataFrame({'user_id':test.user_id.unique()})
    testing['anime_recall'] = testing['user_id'].apply(old_user_user_new_recall, user_item=user_item, W=W, K1=K1, K2=K2)
    return testing



test = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test.pkl') 
user_item = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\user_item.pkl') 
with open(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\old_new_cosine_dist_maxtrix.json','r', encoding='UTF-8') as f:
    W = json.load(f)

start = time.time()
test_old_user_new_recall = get_test_old_user_new_recall(test=test, user_item=user_item, W=W, K1=10, K2=20)
end = time.time()
print('time used:',end-start)

# 把召回的动画id拆分成多行(这样一个user_id就会占多行)
def explode(df,group_by_field,explode_field):
    '''
    @group_field: 分组键, string形式
    @explode_field: 根据df的explode_field列拆分成多行, string形式
    '''
    
    df1 = df[explode_field].apply(pd.Series).stack().rename(explode_field)
    df2 = df1.to_frame().reset_index(1,drop=True)
    res = df2.join(df[group_by_field]).reset_index(drop=True)
    return res.loc[:,[group_by_field,explode_field]]

test_old_user_new_recall = explode(df = test_old_user_new_recall, group_by_field = 'user_id', explode_field = 'anime_recall')
test_old_user_new_recall['recall_channel'] = 'cosine_dist'   # 添加一列召回渠道
test_old_user_new_recall['anime_recall'] = test_old_user_new_recall['anime_recall'].astype(int)
test_old_user_new_recall.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test_old_user_new_recall.pkl')


## 为新用户(即2017-07-01后注册的,或者是训练集中没有喜欢信息的用户)推荐新番
## 原理是：利用之前的用户冷启动召回信息,根据与新用户的性别与年龄最喜欢看的动漫出发,找到和这些动漫相似度最高的
## 新番,然后推荐给新用户

def new_user_new_recall(user_id, gender_age_recall, W, K1=10, K2=20):
    # 用户在冷启动时通过gender_age的混合推荐被推荐的结果，通常有20部动漫
    cold_start = set(gender_age_recall.loc[gender_age_recall.user_id==user_id,'anime_recall'])   
    rec_result = {}
    # a_id是某部通过冷启动人口统计学推荐来的anime_id(也是旧番)
    for a_id in cold_start:
        # 寻找当前动漫(也是旧番)和其他新番的相似系数, 注意下面的矩阵W如果是读取的本地json, 那key变成了string形式, 需要用str转换
        # 找到和当前动漫相似度最高的K1本动漫
        for j, wj in sorted(W[str(a_id)].items(), key=lambda x:x[1], reverse=True)[:K1]: 
            rec_result.setdefault(int(j), 0)
            rec_result[int(j)] += wj
    # 返回权重最高的K2本动漫     
    top_K2_new_anime = [i for i, w in sorted(rec_result.items(), key=lambda x:x[1], reverse=True)[:K2]]
    return top_K2_new_anime


def get_test_new_user_new_recall(test, old_user_recall, gender_age_recall, W, K1=10, K2=20):
    '''
    这个函数是为了给测试集里的新用户(2017-07-01以后注册的),或者是在训练集里没有喜欢记录的“老用户”来推荐新番
    @test: 测试集
    @old_user_recall: 第一步的新番召回结果.其中的用户全是在训练集里有喜欢记录的“老用户”
    @gender_age_recall: 用户冷启动--性别+年龄段的混合召回
    通过新用户的冷启动召回动漫,进而找到和这些动漫最相似的新番,然后推荐给“新用户”
    '''
    # 先挑选出所谓的新用户：
    df = pd.DataFrame({'user_id':list(set(test.user_id) - set(old_user_recall.user_id))})
    df['anime_recall'] = df['user_id'].apply(new_user_new_recall, gender_age_recall=gender_age_recall, W=W, K1=K1, K2=K2)
    return df

test_gender_age_recall = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test_gender_age_recall.pkl')
start = time.time()
test_new_user_new_recall = get_test_new_user_new_recall(test=test, old_user_recall=test_old_user_new_recall, gender_age_recall=test_gender_age_recall, W=W, K1=10, K2=20)
end = time.time()
print('time used:',end-start)

# explode
test_new_user_new_recall = explode(df = test_new_user_new_recall, group_by_field = 'user_id', explode_field = 'anime_recall')
test_new_user_new_recall['recall_channel'] = 'cosine_dist'   # 添加一列召回渠道
test_new_user_new_recall['anime_recall'] = test_new_user_new_recall['anime_recall'].astype(int)
test_new_user_new_recall.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test_new_user_new_recall.pkl')

## 把新用户和老用户的新番推荐合并在一起!
test_new_recall = pd.concat([test_old_user_new_recall, test_new_user_new_recall], axis = 0) \
                    .reset_index(drop = True)
test_new_recall.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test_new_recall.pkl')






  