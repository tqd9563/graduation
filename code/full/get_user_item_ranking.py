# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 08:28:01 2019

@author: tqd95
"""

import pandas as pd
import time

df = pd.DataFrame({'uid':['a','a','b','b','b','a'],
                   'aid':['101','102','201','202','203','103'],
                   'score':[10,7,6,8,5,8]})
tmp = df.groupby('uid') \
        .apply(lambda x: x.sort_values(by='score', ascending=False))['aid'] \


def get_user_item_ranking(train):
    '''
    @train: 非零打分训练集
    统计的是用户在训练集里喜欢的动漫的打分排序
    输出的是一个series,其中index就是user_id, value是一个dict,形如:{anime_id: my_score}
    '''
    res = train.loc[train.label == 1, ['user_id', 'anime_id', 'my_score']] \
                .groupby('user_id') \
                .apply(lambda x: x.sort_values(by='my_score', ascending=False)) \
                .loc[:,['anime_id','my_score']]
    res.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\user_item_ranking.pkl')        
    return


train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.pkl')
start = time.time()     
get_user_item_ranking(train_non_zero)
end = time.time()
print('time used:', (end-start)/60, 'minutes')    




