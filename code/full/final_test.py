# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 10:43:08 2019
生成最终的训练集与测试集
@author: tqd95
"""


import pandas as pd
from tqdm import tqdm
import time

# 读取画像
train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.pkl') # 包含1236的非零分以及4
user_stats = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\user_stats.pkl')
anime_stats = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\anime_stats.pkl')

# 把初始训练集按照日期生序排序后存回本地csv文件
## 注意：下面的写法把排序后打乱的index也给写入到了csv中,即：csv文件的第一列是'Unnamed: 0'
train_non_zero = train_non_zero.sort_values(by='last_update_date',ascending=True)
train_non_zero.to_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.csv')


# 最终训练集(135列)
## 去掉不用的列,合并上画像,最终得到完整版的训练集
def get_train_fn(train, user_stats, anime_stats):
    '''
    @train: 就是train_non_zero,即1236的非零分+4的全部打分记录
    @user_stats: 用户画像
    @anime_stats: 动漫画像
    '''
    df = pd.merge(train, user_stats, on = 'user_id', how = 'left')
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


# 分块读取train_non_zero，生成最终训练集
# 首先生成一个表头
df = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.csv',
                 usecols = ['user_id', 'anime_id', 'label'],
                 nrows=10000)
train = get_train_fn(df, user_stats, anime_stats)
df = pd.DataFrame(columns = train.columns)
df.to_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_fn.csv')

# 然后循环分块写入csv中
## 因为train_non_zero.csv的第一列是'Unnamed: 0', 忘记去掉了, 所以最后保存的train_fn.csv也有这一列。。
## train_fn.csv一共135 + 1 = 136列
def write_train_fn_to_csv():
    chunkSize = 600000
    reader = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.csv',
                             chunksize = chunkSize,
                             usecols = ['user_id', 'anime_id', 'label'])
    start = time.time()
    print('start writing...')
    for df in tqdm(reader):
        train = get_train_fn(df, user_stats, anime_stats)
        train.to_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_fn.csv',
                     header = False, mode = 'a')
    end = time.time()
    print('writing completed!')
    print('time used:', (end-start)/60, 'minutes')
    return

write_train_fn_to_csv()

###############################
# 下面是生成最终测试集
# 读取各路召回结果并合并
test = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test.pkl') 
test_recall = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test_recall.pkl')
test_related_recall = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test_related_recall_10.pkl')
test_gender_age_recall = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test_gender_age_recall.pkl')
test_new_recall = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test_new_recall.pkl')

test_fn = pd.concat([test_recall, test_related_recall, 
                     test_gender_age_recall,test_new_recall], axis = 0) \
            .reset_index(drop = True) \
            .groupby(['user_id','anime_recall']) \
            .apply(lambda x:list(x.recall_channel)) \
            .reset_index() \
            .rename(columns = {0:'recall_channel'})

a = anime_stats.loc[:,['a_anime_id','a_start']]
cond = a['a_start'] > '2017-06-30'
a.loc[cond,'is_new_anime'] = 1
a.loc[-cond,'is_new_anime'] = 0
test_fn = pd.merge(test_fn, a, left_on = 'anime_recall', right_on = 'a_anime_id',how = 'left') \
            .drop(['a_anime_id','a_start'], axis = 1)

test_fn.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test_recall_channel_mapping.pkl')
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
test_fn.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test_fn.pkl')
