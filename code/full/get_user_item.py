# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:58:58 2019
倒排表，过滤动画
@author: tqd95
"""

import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns

# 读取非零打分训练集,序列化为pickle文件
train_non_zero = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.csv')
train_non_zero = train_non_zero.set_index('Unnamed: 0')
train_non_zero.index.name = ''
train_non_zero.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.pkl')

# 用户动漫倒排表
def get_user_item(train):
    user_item = train.loc[train.label == 1, :].groupby('user_id').agg(lambda x: set(x.anime_id))
    user_item = user_item.iloc[:, 0]
    user_item = pd.DataFrame(user_item)
    user_item = user_item.reset_index()
    user_item['num'] = user_item['anime_id'].apply(lambda x: len(x))
    return user_item

user_item = get_user_item(train_non_zero)
user_item.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\user_item.pkl')


watched_anime_set = reduce(lambda x, y: x.union(y), user_item.anime_id)
print('用户动漫正排表里有%d本动画' % len(watched_anime_set))  # 5763本动画


def get_watch_anime_freq(train):
    anime_clean = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\anime_cleaned.csv')
    anime_clean['start'] = anime_clean['aired'].apply(lambda x: eval(x)['from'])
    anime_clean['end'] = anime_clean['aired'].apply(lambda x: eval(x)['to'])
    watched_anime_freq = train.loc[train.label == 1, :].groupby('anime_id').size()
    watched_anime_freq = watched_anime_freq.reset_index()
    watched_anime_freq = watched_anime_freq.rename(columns={0: 'times'})
    watched_anime_freq = pd.merge(watched_anime_freq, anime_clean, on='anime_id')
    watched_anime_freq = watched_anime_freq.loc[:, ['anime_id', 'times', 'title', 'score', 'start', 'end', 
                                                    'rank', 'popularity', 'favorites']]
    return watched_anime_freq

watched_anime_freq = get_watch_anime_freq(train_non_zero)
watched_anime_freq.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\watched_anime_freq.pkl')


num_quantile = pd.DataFrame({'q': list(map(lambda x: x / 100, range(90, 100, 1)))})
num = []
for i in range(90, 100, 1):
    num.append(user_item['num'].quantile(i / 100))
num_quantile['num'] = num


fig, ax = plt.subplots(1, 1)
plt.rcParams['font.sans-serif'] = ['SimHei']
sns.lineplot(x='times', y='score', data=watched_anime_freq.query('times<=100'))
ax.set_title('在训练集中y=1并且观看次数小于50次的动画，其平均得分趋势')