# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:43:37 2019
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
    t = 1
    for index, row in user_item.iterrows():
        if t % 500 == 1:
            print('epoch:', t, '/', 103479, '...')
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
        t += 1
    end = time.time()
    print('time used:', (end - start)/60, 'minutes')
    return item_user_count, item_co_count


user_item = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\user_item.pkl')
item_user_count, item_co_count = get_count_matrix(user_item)  # 24.6min
# 存入本地
with open(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\item_co_count.json', 'w') as outfile:
    json.dump(item_co_count, outfile)

with open(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\item_user_count.json', 'w') as outfile:
    json.dump(item_user_count, outfile)



# 计算动画间相似矩阵
def get_similarity_matrix(item_user_count, item_corelated_count):
    '''
    @item_user_count: 每本动漫被多少用户喜欢看, 是一个一维的dict
    @item_corelated_count: 每本动漫和其他动漫共同被喜欢看的用户数, 是一个二维嵌套的dict
    输出的相似矩阵W, 也是二维嵌套dict, 其内外层的key不再是动漫对应的index,而是直接用anime_id
    '''
    W = dict()

    # 对于每一部动漫，只保留和他相似度最高的500本动漫
    start = time.time()
    t = 1
    for i, related_items in item_corelated_count.items():
        if t % 100 == 1:
            print('epoch:', t, '/', 5763, '...') 
        for j, cij in related_items.items():
            W.setdefault(int(i), {})
            W[int(i)].setdefault(int(j), 0)
            W[int(i)][int(j)] = cij / np.sqrt(item_user_count[i] * item_user_count[j])
        # 如果超出300本，则去掉和当前动漫i相似度最低的那部动漫
        if len(W[int(i)]) > 300:            
            W[int(i)] = dict(sorted(W[int(i)].items(),key=lambda x:x[1], reverse=True)[:300])
        t += 1
    end = time.time()
    print('time used:', (end - start)/60, 'minutes')
    return W


with open(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\item_co_count.json', 'r', encoding='UTF-8') as f1:
    item_co_count = json.load(f1)
with open(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\item_user_count.json', 'r', encoding='UTF-8') as f2:
    item_user_count = json.load(f2)
    
W = get_similarity_matrix(item_user_count, item_co_count)  # 这个W是用的anime_id作为Key的

# 把相似度矩阵保存到本地Json文件
# 注意：保存的json文件再读取到python中时, 其内外层的key都变成了string类型！
with open(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\similarity_maxtrix_top300.json', 'w') as outfile:
    json.dump(W, outfile)
    
    