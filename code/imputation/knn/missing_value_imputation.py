# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:29:37 2019
0分填充策略--KNN
@author: tqd95
"""

import pandas as pd
import numpy as np
import time
from sklearn.neighbors import KNeighborsRegressor

train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_non_zero.pkl')  # 包含1236的非零分以及4
train_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_zero.pkl')
anime_clean = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\anime_cleaned.csv')
user_stats = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\user_stats.pkl')
anime_stats = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\anime_stats.pkl')


# 先把原始数据集里，episodes=0的动漫，用观看过的最大集数代替
# train_zero和train_non_zero都要替换
tmp1 = train_non_zero.query('episodes==0').groupby(['anime_id','title'])['my_watched_episodes'] \
                    .agg([('min',np.min),('episodes',np.max)]) \
                    .reset_index()[['anime_id','episodes']]
tmp2 = train_zero.query('episodes==0').groupby(['anime_id','title'])['my_watched_episodes'] \
                    .agg([('min',np.min),('episodes',np.max)]) \
                    .reset_index()[['anime_id','episodes']]
tmp = pd.merge(tmp1, tmp2, on = 'anime_id', how = 'outer')
tmp.fillna(0, inplace = True)
tmp['value_flag'] = tmp['episodes_x'] - tmp['episodes_y'] 
tmp_bigger = tmp[tmp['value_flag'] >= 0].copy()
tmp_smaller = tmp[tmp['value_flag'] < 0].copy()
tmp_bigger['episodes'] = tmp_bigger['episodes_x']
tmp_smaller['episodes'] = tmp_smaller['episodes_y']
tmp = pd.concat([tmp_bigger.loc[:,['anime_id','episodes']], tmp_smaller.loc[:,['anime_id','episodes']]],axis = 0) \
        .reset_index() \
        .drop('index',axis = 1)

# 28921, 33865, 35443

def episodes_transform(df, tmp):
    t = pd.merge(df, tmp, on = 'anime_id', how='left')
    t['episodes_y'].fillna(0, inplace = True)
    t['episodes'] = t.episodes_x + t.episodes_y
    del t['episodes_x']
    del t['episodes_y']
    int_columns = t.select_dtypes(include=['int64']).columns
    t[int_columns] = t[int_columns].apply(pd.to_numeric,downcast='unsigned') 
    return t

train_zero = episodes_transform(train_zero, tmp)
train_zero = train_zero[['user_id', 'anime_id', 'my_watched_episodes' ,'episodes', 'my_score', 
                         'my_status', 'last_update_date', 'title', 'score', 'rank', 'start', 'end']]
train_non_zero = episodes_transform(train_non_zero, tmp)
train_non_zero = train_non_zero[['user_id', 'anime_id', 'my_watched_episodes' ,'episodes', 'my_score', 'my_status', 
                                 'last_update_date', 'title', 'score', 'rank', 'start', 'end', 'label']]

# 选出观看集数占总集数比例大于等于50%的样本, 这部分样本的score缺失值会被预测(大约有69%), 命名为target_zero
# 把这个target_zero转化成最终训练集的形式
cond = train_zero.my_watched_episodes / train_zero.episodes >= 0.5
target_zero = train_zero.loc[cond, :]
target_zero.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\imputation\target_zero.pkl')


def get_train_fn(train, user_stats, anime_stats):
    '''
    @train: 就是target_zero以及train_non_zero
    @user_stats: 用户画像
    @anime_stats: 动漫画像
    '''
    df = train.loc[:,['user_id','anime_id']]
    df = pd.merge(df, user_stats, on = 'user_id', how = 'left')
    df = pd.merge(df, anime_stats, left_on = 'anime_id', right_on = 'a_anime_id', how = 'left')
    df = df.drop(['user_id','anime_id','a_anime_id','join_date','a_start','a_end','a_studio'],axis=1)
    # 把类别列one-hot连续化
    object_columns = df.select_dtypes(include=['object']).columns
    for col in object_columns:
        df = pd.concat([df, pd.get_dummies(df[col], prefix = col)], axis = 1)
        df.drop(col, axis = 1, inplace = True)    
    df['a_popularity'] = df['a_popularity'].map({'Unpopular':1,'Not so popular':2,'Common':3,
                                                  'A bit popular':4,'Quite popular':5,'Extremely popular':6})
    # 对float列进行downcast处理
    col_float = df.select_dtypes(include=['float64']).columns
    df[col_float] = df[col_float].apply(pd.to_numeric, downcast = 'float')
    # 把一些错变成float的列转换回int,然后对所有的int也进行downcast处理
    df.loc[:,['u_watching', 'u_completed', 'u_on_hold','u_dropped', 'u_liked']] = df.loc[:,['u_watching', 'u_completed', 'u_on_hold','u_dropped', 'u_liked']].astype(int)
    col_int = df.select_dtypes(include=['int32','int64']).columns 
    df[col_int] = df[col_int].apply(pd.to_numeric, downcast = 'unsigned')
    return df

start = time.time()
x_train = get_train_fn(train_non_zero, user_stats, anime_stats)
x_test = get_train_fn(target_zero, user_stats, anime_stats)
y_train = train_non_zero['my_score']
end = time.time()
print('time used:', (end-start)/60, 'minutes')  # 0.49 minutes

# 用KNN来填补缺失值
def missing_score_imputation(x_train, y_train, test, k = 5):
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1)
    knn.fit(x_train, y_train)
    return knn.predict(test)

start = time.time()
predict_score = missing_score_imputation(x_train, y_train, x_test)
end = time.time()
print('time used:', (end-start)/60, 'minutes')  # 0.55 minutes

# 把预测的得分合并到target_zero中, 并覆盖原来的my_score
predict_score = pd.DataFrame(predict_score).rename(columns = {0: 'my_score'})
predict_score.index = target_zero.index
target_zero.drop('my_score', axis = 1, inplace = True)
target_zero = pd.concat([target_zero, predict_score], axis = 1)
target_zero = target_zero[['user_id', 'anime_id', 'my_watched_episodes', 'episodes', 'my_score', 'my_status',
                           'last_update_date', 'title', 'score', 'rank', 'start', 'end']]

# 根据原有的label生成规则, 生成新的label, 同时添加weight = my_watched_episodes / episodes
## 最后把target_zero合并到原来的train_non_zero中去, 成为新的训练集骨架！
target_zero.loc[target_zero.my_score >= 8, 'label'] = 1
target_zero.loc[target_zero.my_score < 8, 'label'] = 0
target_zero['confidence_weight'] = target_zero['my_watched_episodes'] / target_zero['episodes']
train_non_zero['confidence_weight'] = train_non_zero['my_watched_episodes'] / train_non_zero['episodes']
train_non_zero = pd.concat([train_non_zero, target_zero], axis = 0)

train_non_zero.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\imputation\knn\train_non_zero.pkl')

# 生成带权的新的user_item用户-物品倒排表
def get_user_item(train):
    user_item = train.loc[train.label == 1, :].groupby('user_id') \
                    .apply(lambda df:dict(zip(df.anime_id,df.confidence_weight))) \
                    .to_frame().reset_index() \
                    .rename(columns = {0: 'anime_id'})
    user_item['num'] = user_item['anime_id'].apply(lambda x: len(x))
    return user_item

user_item = get_user_item(train_non_zero)
user_item.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\imputation\knn\user_item.pkl')




# 在user_item里添加权重信息
df2 = pd.DataFrame({'user_id':['11','11','11','22','22'],
                    'aid':[1,4,5,1,3],
                    'weight':[1,1,0.5,0.5,1]})

df2.groupby('user_id').apply(lambda df:dict(zip(df.aid,df.weight)))
