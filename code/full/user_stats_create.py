# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:25:58 2019
生成用户画像特征
@author: tqd95
"""

import numpy as np
import pandas as pd
import time

user_clean = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\users_cleaned.csv')
anime_stats = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\anime_stats.pkl')
train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.pkl')

t1 = train_non_zero.loc[:,['user_id','anime_id','my_watched_episodes','my_score','my_status',
                           'label','last_update_date']]

t1 = pd.merge(t1, anime_stats, left_on = 'anime_id', right_on = 'a_anime_id')
t1.drop('a_anime_id', axis = 1, inplace = True)  

# u1就是我们的用户画像骨架, u1的user_id就是对用户表采样5%后得到的5436名用户,
# 里面包含了t1的全部用户、train_plan_to_watch的全部用户，以及test里的用户
u1 = user_clean.loc[:,['user_id','gender','birth_date','join_date']] # 没要location,我觉得不好用...

# 以训练/测试集的划分界限为当前时刻,计算用户的年龄
def transfer_age(dt):
    dt1 = pd.to_datetime('2017-06-30')
    age =  (dt1 - pd.to_datetime(dt))/np.timedelta64(1,'Y')
    return int(1 + np.floor(age))

u1['age'] = u1['birth_date'].apply(transfer_age)
u1.drop('birth_date', axis = 1, inplace = True)


# 每个user的各status频数统计
def get_user_status_stats(train):
    '''
    @train: 用户打分行为join上动漫画像后的中间训练表,不包含打分为0的(dropped除外)
    '''
    u = pd.DataFrame(train.user_id.unique()).rename(columns = {0:'user_id'})   # 训练集中的所有user_id
    mapping = {1:'u_watching',2:'u_completed',3:'u_on_hold',4:'u_dropped'}
    for status in range(1,5):
        tmp = train.loc[train.my_status == status, ['user_id','anime_id']]
        res = tmp.groupby('user_id') \
             .size().sort_values(ascending = False) \
             .rename(mapping[status]) \
             .reset_index()
        res[mapping[status]].fillna(0, inplace = True)
        res[mapping[status]] = res[mapping[status]].apply(int)
        u = pd.merge(u, res, on = 'user_id', how = 'left')
        u.fillna(0, inplace = True)    
    return u

start = time.time()
u1 = pd.merge(u1, get_user_status_stats(t1), on = 'user_id', how = 'left')
u1.fillna(0, inplace = True)
end = time.time()
print('time used:', (end-start)/60, 'minutes')

# 统计每个用户在训练集里喜欢了多少动漫(label=1)
u2 = t1.query('label==1').groupby('user_id').size().rename('u_liked').reset_index()
u1 = pd.merge(u1,u2,on='user_id',how='left')    # 此处用left join, 有少部分用户虽然在训练集里出现, 但是没有一部label=1的动漫。。
u1.fillna(0, inplace = True)


# 根据动漫分类的汇总信息(只统计每个用户在label=1条件下的各类型动漫比例)
def get_user_liked_pct(train,fields):
    '''
    @train: 用户打分行为join上动漫画像后的中间训练表,不包含打分为0的(dropped除外)
    @fields: 根据哪个动漫特征来进行分组统计, 是string类型
    '''
    group_list = train[fields].unique()
    u = pd.DataFrame(train.user_id.unique()).rename(columns = {0:'user_id'})   # 训练集中的所有user_id
    u1 = train.query('label==1').groupby('user_id').size().rename('u_liked').reset_index()
    u = pd.merge(u, u1, on = 'user_id', how = 'left')   # 给每个用户添加上他们label=1的动漫数
    for s in group_list:
        tmp = train.loc[((train[fields] == s) & (train.label==1)),['user_id','anime_id']]        
        col_name = 'u_' + s + '_liked'
        res = tmp.groupby('user_id') \
                .size().rename(col_name) \
                .reset_index()
        u = pd.merge(u, res, on = 'user_id', how = 'left')
        u.fillna(0, inplace = True)
        # 计算对应source的喜欢比例(该用户label=1的动漫中, source是这个的比例)
        col_name_2 = 'u_' + s + '_liked_pct'
        u[col_name_2] = u[col_name]/u['u_liked']
        u.fillna(0, inplace = True)
        u.drop(col_name, axis = 1, inplace = True)
    u.drop('u_liked', axis = 1, inplace = True)
    return u

## 分别对source, rating统计各类占比
u1 = pd.merge(u1, get_user_liked_pct(t1,'a_source'), on = 'user_id', how = 'left')
u1 = pd.merge(u1, get_user_liked_pct(t1,'a_rating'), on = 'user_id', how = 'left')
u1.fillna(0, inplace = True)

## 对genre要单独处理：
def get_user_genre_liked_pct(train):
    '''
    @train: 用户打分行为join上动漫画像后的中间训练表,不包含打分为0的(dropped除外)
    '''
    tmp = train.query('label==1') \
            .filter(regex='(^a_genre)|(^user_id)',axis=1) \
            .groupby('user_id') \
            .sum() \
            .applymap(int) \
            .reset_index()
    u1 = pd.DataFrame(train.user_id.unique()).rename(columns = {0:'user_id'})     
    u2 = train.query('label==1').groupby('user_id').size().rename('u_liked').reset_index()
    u1 = pd.merge(u1, u2, on='user_id', how='left')   
    u1.fillna(0, inplace = True)
    res = pd.merge(u1, tmp, on = 'user_id', how = 'left')
    res.fillna(0, inplace = True)
    for col in res.columns[2:]:
        new_colname = 'u_' + col[2:] + '_liked_pct'    # 新的百分比统计量的列名
        res[new_colname] = res[col]/res['u_liked']
        res.fillna(0, inplace = True)
        res.drop(col, axis = 1, inplace = True)
    res.drop('u_liked', axis = 1, inplace = True)
    return res

## 把genre的分类统计结果join到u1上
u1 = pd.merge(u1, get_user_genre_liked_pct(t1), on = 'user_id', how = 'left')
u1.fillna(0, inplace = True)


# 下面看的是之前单独分开来的,原始评分行为数据中,所有status = 6(plan to watch)的样本
# 目标是把这些用户打算观看但是没看的动漫信息, 抽象成用户特征
# 之前的训练数据生成的用户画像大多是百分比统计量,这里考虑是继续用百分比呢？还是直接选top1,top3之类的
# 还是用百分比吧。。

train_plan_to_watch = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_plan_to_watch.pkl')
train_plan_to_watch = pd.merge(train_plan_to_watch, anime_stats, left_on = 'anime_id', right_on = 'a_anime_id')
train_plan_to_watch.drop('a_anime_id',axis = 1,inplace = True)


def get_user_genre_planned_pct(train):
    '''
    @train: 用户plan to watch的数据集
    '''
    tmp = train.filter(regex='(^a_genre)|(^user_id)',axis=1) \
            .groupby('user_id') \
            .sum() \
            .applymap(int) \
            .reset_index()
    u1 = pd.DataFrame(train.user_id.unique()).rename(columns = {0:'user_id'})     
    u2 = train.groupby('user_id').size().rename('u_planned').reset_index()
    u1 = pd.merge(u1, u2, on='user_id', how='left')   
    u1.fillna(0, inplace = True)
    res = pd.merge(u1, tmp, on = 'user_id', how = 'left')
    res.fillna(0, inplace = True)
    for col in res.columns[2:]:
        new_colname = 'u_' + col[2:] + '_planned_pct'    # 新的百分比统计量的列名
        res[new_colname] = res[col]/res['u_planned']
        res.fillna(0, inplace = True)
        res.drop(col, axis = 1, inplace = True)
    res.drop('u_planned', axis = 1, inplace = True)
    return res


u1 = pd.merge(u1, get_user_genre_planned_pct(train_plan_to_watch), on = 'user_id', how = 'left')
u1.fillna(0, inplace = True)

def get_user_planned_pct(train,fields):
    '''
    @train: 用户plan to watch的数据集
    @fields: 根据哪个动漫特征来进行分组统计, 是string类型
    '''
    group_list = train[fields].unique()
    u = pd.DataFrame(train.user_id.unique()).rename(columns = {0:'user_id'})   # 训练集中的所有user_id
    u1 = train.groupby('user_id').size().rename('u_planned').reset_index()
    u = pd.merge(u, u1, on = 'user_id', how = 'left')   # 给每个用户添加上他们label=1的动漫数
    for s in group_list:
        tmp = train.loc[train[fields] == s,['user_id','anime_id']]        
        col_name = 'u_' + s + '_planned'
        res = tmp.groupby('user_id') \
                .size().rename(col_name) \
                .reset_index()
        u = pd.merge(u, res, on = 'user_id', how = 'left')
        u.fillna(0, inplace = True)
        # 计算对应source的喜欢比例(该用户label=1的动漫中, source是这个的比例)
        col_name_2 = 'u_' + s + '_planned_pct'
        u[col_name_2] = u[col_name]/u['u_planned']
        u.fillna(0, inplace = True)
        u.drop(col_name, axis = 1, inplace = True)
    u.drop('u_planned', axis = 1, inplace = True)
    return u

u1 = pd.merge(u1, get_user_planned_pct(train_plan_to_watch,'a_source'), on = 'user_id', how = 'left')
u1 = pd.merge(u1, get_user_planned_pct(train_plan_to_watch,'a_rating'), on = 'user_id', how = 'left')
u1.fillna(0, inplace = True)

u1.rename(columns = {'age':'u_age','gender':'u_gender'},inplace = True)
u1.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\user_stats.pkl')
