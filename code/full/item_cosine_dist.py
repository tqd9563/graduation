# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:46:13 2019
求老番与新番间的cosine距离
@author: tqd95
"""

import numpy as np
import pandas as pd
import time
import json


anime_clean = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\anime_cleaned.csv')
anime_stats = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\anime_stats.pkl')
train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.pkl') # 包含1236的非零分以及4

# 考虑新番的推荐,即上映日期在2017-06-30以后的动漫
## 现采用的方法是计算每一部新番和老番之间的余弦相似度(基于标签的推荐),然后对每个用户在训练集里喜欢过的动漫,
## 找到它和每一部新番的相似度,累加;最后对所有新番对该用户的相似度降序排序,然后取前K个进行召回
## 用到的特征主要有：20维的genre信息, rating的2个稀有分级如Rx-Hentai,R+ - Mild Nudity, 以及9维的source信息 
def get_anime_genre(total_anime_list):
    '''
    @total_anime_list: 全动漫信息
    返回结果: 每个anime_id对应的genre,一个anime_id有多行
    '''
    g = total_anime_list.loc[:,['anime_id','genre']]
    g['genre'].fillna('',inplace=True)
    genre = g['genre'].apply(lambda x:x.split(',')).apply(pd.Series).stack().rename('genre')
    genre = genre.to_frame().reset_index(1,drop=True)
    genre['genre'] = genre['genre'].apply(lambda s:s.strip())
    genre = genre.join(g.anime_id).reset_index(drop=True)
    anime_genre = genre.loc[:,['anime_id','genre']]
    return anime_genre


# 建立feature-item倒排表：
def get_feature_item(total_anime_list, anime_stats):
    '''
    @total_anime_list: 原始的动漫信息表,为了获取genre
    @anime_stats: 动漫画像表
    输出的列feature,包含了20个genre和2个rating
    '''
    a = get_anime_genre(total_anime_list)
    top_20_genre = anime_stats.filter(regex = 'a_genre').columns
    top_20_genre = [g[8:] for g in top_20_genre]
    top_20_genre.pop(top_20_genre.index('Other'))
    cond = a['genre'].apply(lambda x:x not in top_20_genre)
    a.loc[cond,'genre'] = 'Other'
    res1 = a.groupby('genre') \
            .agg(lambda x: set(x.anime_id)) \
            .reset_index() \
            .rename(columns = {'genre':'feature'}) \
            .query('feature != "Other"')
    res2 = total_anime_list.query('rating=="R+ - Mild Nudity" or rating=="Rx - Hentai"') \
            .loc[:,['anime_id','rating']] \
            .groupby('rating') \
            .agg(lambda x: set(x.anime_id)) \
            .reset_index() \
            .rename(columns = {'rating':'feature'})
    res3 = anime_stats.loc[:,['a_anime_id','a_source']] \
            .groupby('a_source') \
            .agg(lambda x: set(x.a_anime_id)) \
            .reset_index() \
            .rename(columns = {'a_source':'feature','a_anime_id':'anime_id'})
    res = pd.concat([res1,res2,res3],axis = 0).reset_index(drop=True)
    res['num'] = res['anime_id'].apply(lambda x:len(x))    
    return res


feature_item = get_feature_item(anime_clean, anime_stats)
feature_item.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\feature_item.pkl')

# 生成物品共现矩阵
def get_count_matrix(feature_item):  
    '''
    @feature_item: 特征-动漫倒排表，每个feature包括了哪些动漫(新番和旧番都在)
    返回结果：
    @item_feature_count: dict{anime_id: genre_counts}：每部动漫占据多少个feature的tag
    @item_co_count: dict{anime_id1: {}, anime_id2:{}, ...}：每本动漫分别和其他动画的tag交集人数，二维嵌套dict
    '''
    item_feature_count = dict() 
    item_co_count = dict() 
    start = time.time()
    for index,row in feature_item.iterrows():
        anime_list = row['anime_id']
        for i in anime_list:
            item_feature_count.setdefault(i,0)
            item_feature_count[i] += 1
            for j in anime_list:
                if j == i: continue
                item_co_count.setdefault(i,{})
                item_co_count[i].setdefault(j,0)
                item_co_count[i][j] += 1
    end = time.time()
    print('用时：',end-start)
    return item_feature_count, item_co_count

item_feature_count, item_co_count = get_count_matrix(feature_item) 


# 生成老番-新番间余弦距离矩阵
def get_old_new_cosine_dist_matrix(train, anime_stats, item_feature_count, item_co_count):
    '''
    @train: status = 1236的非零打分记录以及status = 4的所有打分记录
    @anime_stats: 动漫画像, 为了得到所有的新番id
    @item_feature_count: 每部动漫占据多少个feature的tag,是一维dict
    @item_co_count: 每本动漫分别和其他动画的tag交集人数，二维嵌套dict
    Output：
    行是用户在训练集里喜欢的所有anime_id, 共5511部(全都是旧番)
    列是所有的新番id,共452部
    注意：这和item-CF那次不一样,保存的都是anime_id而不再是index了！！
    '''
    W = dict()
    old_anime_list = set(train.anime_id)
    new_anime_list = set(anime_stats.query('a_start > "2017-06-30"').a_anime_id)
    start = time.time()
    # 因为我们的行只要旧番,所以要先对item_co_count的keys做一个过滤,只留下属于旧番的key(anime_id)
    old_item_co_count = dict(list(filter(lambda x:x[0] in old_anime_list,item_co_count.items())))
    for i, related_items in old_item_co_count.items():
        # i是旧番id, related_items是和这部旧番有公共feature的其他anime_id,并记录其各自共有的feature数目
        # 因为我们的列只要新番,所以先对related_items的keys做一个过滤,只留下属于新番的key(anime_id)
        new_item_co_count = dict(list(filter(lambda x:x[0] in new_anime_list,related_items.items())))
        for j, cij in new_item_co_count.items():
            W.setdefault(i,{})
            W[i].setdefault(j,0)
            W[i][j] = cij/np.sqrt(item_feature_count[i] * item_feature_count[j])
    end = time.time()
    print('time used:', end-start)
    return W

W = get_old_new_cosine_dist_matrix(train_non_zero, anime_stats, item_feature_count, item_co_count)

# 把余弦距离矩阵保存到本地Json文件
# 注意：保存的json文件再读取到python中时, 其内外层的key都变成了string类型！
with open(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\old_new_cosine_dist_maxtrix.json','w') as outfile:
    json.dump(W, outfile)


