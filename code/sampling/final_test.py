# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:41:23 2019
生成最终的训练集与测试集
@author: tqd95
"""

import pandas as pd
# 读取画像
train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_non_zero.pkl') # 包含1236的非零分以及4
user_stats = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\user_stats.pkl')
anime_stats = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\anime_stats.pkl')

# 读取测试集召回表
test = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test.pkl') 
test_recall = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test_recall.pkl')
test_related_recall = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test_related_recall_10.pkl')
test_gender_age_recall = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test_gender_age_recall.pkl')
test_new_recall = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test_new_recall.pkl')

# 最终训练集(135列)
## 去掉不用的列,合并上画像,最终得到完整版的训练集
def get_train_fn(train, user_stats, anime_stats):
    '''
    @train: 就是train_non_zero,即1236的非零分+4的全部打分记录
    @user_stats: 用户画像
    @anime_stats: 动漫画像
    '''
    df = train.loc[:,['user_id','anime_id','label']]
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


train_fn = get_train_fn(train_non_zero, user_stats, anime_stats)
train_fn.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_fn.pkl')

# 占用内存大小：从994.6MB减小到了324.5MB
train_fn.info(memory_usage='deep')  




t = test.query('my_watched_episodes>0 and is_new == 0').groupby('user_id').size()
cnt_quantile = pd.DataFrame({'q':list(map(lambda x:x/10,range(1,10)))})
cnt = []
for i in range(1,10):
    cnt.append(t.quantile(i/10))
cnt_quantile['cnt']=cnt  


## 合并各类recall,去重,保留recall_channel的情况下保存一份副本,供模型预测结果分析时的对照
## 注意加上召回的动漫是新番还是旧番的标记
test_fn = pd.concat([test_recall, test_related_recall, 
                     test_gender_age_recall,test_new_recall], axis = 0) \
            .reset_index(drop = True) \
            .groupby(['user_id','anime_recall']) \
            .apply(lambda x:list(x.recall_channel)) \
            .reset_index() \
            .rename(columns = {0:'recall_channel'})
            
# 下面这部分可以不做，直接生成最终测试集           
a = anime_stats.loc[:,['a_anime_id','a_start']]
cond = a['a_start'] > '2017-06-30'
a.loc[cond,'is_new_anime'] = 1
a.loc[-cond,'is_new_anime'] = 0
test_fn = pd.merge(test_fn, a, left_on = 'anime_recall', right_on = 'a_anime_id',how = 'left') \
            .drop(['a_anime_id','a_start'], axis = 1)

test_fn.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test_recall_channel_mapping.pkl')
test_fn.drop('is_new_anime', axis=1, inplace=True)  # 记录下状态后要删掉这列,因为不会进模型的


# 最终测试集(134列)
def get_test_fn(test, user_stats, anime_stats):
    '''
    @test: 测试集,只有三列,分别是user_id, anime_id和recall_channel
     @user_stats: 用户画像
    @anime_stats: 动漫画像
    '''
    df = pd.merge(test, user_stats, on = 'user_id', how = 'left')
    df = pd.merge(df, anime_stats, left_on = 'anime_recall', right_on = 'a_anime_id', how = 'left')
    df = df.drop(['user_id','anime_recall','recall_channel','a_anime_id','join_date','a_start','a_end','a_studio'],axis=1)
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
    df[df.filter(regex='^a_genre').columns] = df[df.filter(regex='^a_genre').columns].astype(int)
    df.loc[:,['u_watching', 'u_completed', 'u_on_hold','u_dropped', 'u_liked']] = df.loc[:,['u_watching', 'u_completed', 'u_on_hold','u_dropped', 'u_liked']].astype(int)
    col_int = df.select_dtypes(include=['int32','int64']).columns 
    df[col_int] = df[col_int].apply(pd.to_numeric, downcast = 'unsigned')
    return df
    
test_fn = get_test_fn(test_fn, user_stats, anime_stats)
test_fn.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test_fn.pkl')
# 占用内存大小：
test_fn.info(memory_usage='deep')
