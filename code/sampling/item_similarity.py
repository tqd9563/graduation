# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 19:03:32 2019
求物品间相似矩阵
@author: tqd95
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import json


# 生成物品共现矩阵
def get_count_matrix(user_item):
    '''
    @user_item: 用户-动漫倒排表，即每个用户喜欢了哪些动漫
    返回结果：
    @item_user_count: dict{anime_id: liked_user_counts}：每部动漫被多少人喜欢看
    @item_co_count: dict{anime_id1: {}, anime_id2:{}, ...}：每本动漫分别和其他动画的交集人数，二维嵌套dict
    '''
    item_user_count = dict()
    item_co_count = dict()
    start = time.time()
    for index, row in user_item.iterrows():
        anime_list = row['anime_id']
        for i in anime_list:
            item_user_count.setdefault(i, 0)
            item_user_count[i] += 1
            for j in anime_list:
                if j == i:
                    continue
                item_co_count.setdefault(i, {})
                item_co_count[i].setdefault(j, 0)
                item_co_count[i][j] += 1
    end = time.time()
    print('用时：', end - start)
    return item_user_count, item_co_count


user_item = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\user_item.pkl')
item_user_count, item_co_count = get_count_matrix(user_item)  # 用时82s

# 计算动画间相似矩阵


def get_similarity_matrix(item_user_count, item_corelated_count):
    '''
    @item_user_count: 每本动漫被多少用户喜欢看, 是一个一维的dict
    @item_corelated_count: 每本动漫和其他动漫共同被喜欢看的用户数, 是一个二维嵌套的dict
    输出的相似矩阵W, 也是二维嵌套dict, 其内外层的key不再是动漫对应的index,而是直接用anime_id
    '''
    W = dict()
    # 构造一个dict映射，key是anime_id, value是动画对应的index, index取值范围是0-1367
    # 内外层循环中的变量i, j都是anime_id
    #mapping = dict(anime_list.anime_id)
    #mapping = {v:k for k,v in mapping.items()}
    start = time.time()
    for i, related_items in item_corelated_count.items():
        for j, cij in related_items.items():
            #index_row = mapping[i]
            #index_col = mapping[j]
            W.setdefault(i, {})
            W[i].setdefault(j, 0)
            W[i][j] = cij / np.sqrt(item_user_count[i] * item_user_count[j])
    end = time.time()
    print('用时：', end - start)
    return W


W = get_similarity_matrix(item_user_count, item_co_count)  # 这个W是用的anime_id作为Key的


# 把相似度矩阵保存到本地Json文件
# 注意：保存的json文件再读取到python中时, 其内外层的key都变成了string类型！
with open(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\similarity_maxtrix.json', 'w') as outfile:
    json.dump(W, outfile)
