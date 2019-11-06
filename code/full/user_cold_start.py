# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:41:46 2019
用户冷启动推荐
@author: tqd95
"""

import numpy as np
import pandas as pd
import time
from tqdm import tqdm

train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.pkl') # 包含1236的非零分以及4

def age_mapping(age):
    '''
    @age: 输入具体的年龄,返回一个string
    '''
    if age < 19:
        return 'Teenager'
    elif age < 22:
        return 'Youth'
    elif age < 26:
        return 'Youth Adult'
    elif age < 30:
        return 'Adult'
    else:
        return 'Middle Aged'
    
# 得到性别+年龄的分组排名
def get_gender_age_ranking(train,gender,age):
    '''
    @train:非零打分训练集(dropped除外)
    @gender: string形式,可取Male,Female(Non-Binary单独处理)
    @age: string形式,有5种取值:'Teenager','Youth','Youth Adult','Adult','Middle Aged'
    '''
    user_stats = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\user_stats.pkl')
    a = pd.merge(train, user_stats.loc[:,['user_id','u_gender','u_age']], on = 'user_id')
    a['age'] = a['u_age'].apply(age_mapping)
    a = a.loc[((a.u_gender == gender) & (a.age == age)), :]
    
    num_of_users = len(set(a.user_id)) # 对应这个年龄段和性别的总人数
    a = a.groupby('anime_id').agg({'my_score':np.mean, 'label':[np.mean,'count']}) 
    a.columns = a.columns.get_level_values(0)
    a.columns = pd.Index(['avg_my_score','avg_liked_pct','cnt'])
    #r = a.query('cnt>=100').reset_index()    # 选出在train_non_zero中记录数超过100的动漫, 防止太过小众 
    # 上面的做法太激进了，现在换成记录数超过这个年龄段+这个性别总人数的30%即可
    r = a.loc[a.cnt >= 0.2*num_of_users,:].reset_index()
    r['score_rank'] = r['avg_my_score'].rank(ascending=False)
    r['liked_pct_rank'] = r['avg_liked_pct'].rank(ascending=False)
    r['avg_rank'] = (r['score_rank'] + r['liked_pct_rank']) / 2
    r = r.sort_values(by = 'avg_rank', ascending = True)
    r['gender'] = gender
    r['age'] = age
    return r



genders = ['Male','Female']
ages = ['Teenager','Youth','Youth Adult','Adult','Middle Aged']

start0 = time.time()
start = start0
gender_age_ranking = pd.DataFrame()
i,j = 1,1
for gender in genders:
    for age in ages:
        print('epoch:', i,'--',j, '/', 5)
        gender_age_ranking = pd.concat([gender_age_ranking, get_gender_age_ranking(train_non_zero,gender,age)],axis=0)
        j += 1
        end = time.time()
        print('time used:', (end-start)/60, 'minutes')
        start = end
    i += 1
end = time.time()
print('total time used:', (end-start0)/60, 'minutes')


gender_age_ranking.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\gender_age_ranking.pkl')



# 为单个测试集用户进行召回
def user_gender_age_recall(user_id, gender_age_ranking, user_watched_item, K=20):
    '''
    @gender_age_ranking: 根据性别以及年龄分段的string,找到对应分段的top动漫,然后加以过滤后召回
    '''
    user_stats = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\user_stats.pkl')
    age = list(user_stats.loc[user_stats.user_id == user_id,:]['u_age'])[0]
    gender = list(user_stats.loc[user_stats.user_id == user_id,:]['u_gender'])[0]
    age_interval = age_mapping(age)
    # 如果性别不是Non-Binary,那简单
    if gender != 'Non-Binary': 
        recall_anime_list = gender_age_ranking.loc[((gender_age_ranking.age == age_interval) & (gender_age_ranking.gender == gender)),:]
        rec = []
        for index, row in recall_anime_list.iterrows():
            a_id = row['anime_id']
            if a_id in user_watched_item.get(user_id,[]):
                continue
            else:
                rec.append(a_id)
            if len(rec) >= K:
                break
        return rec
    # 如果是Non-Binary,那就对应这个年龄分段的Male和Female召回里各取10个出来。
    else:
        recall_anime_list_1 = gender_age_ranking.loc[((gender_age_ranking.age == age_interval) & (gender_age_ranking.gender == 'Male')),:]
        rec = []
        for index, row in recall_anime_list_1.iterrows():
            a_id = row['anime_id']
            if a_id in user_watched_item.get(user_id,[]):
                continue
            else:
                rec.append(a_id)
            if len(rec) >= K//2:
                break
        recall_anime_list_2 = gender_age_ranking.loc[((gender_age_ranking.age == age_interval) & (gender_age_ranking.gender == 'Female')),:]
        for index, row in recall_anime_list_2.iterrows():
            a_id = row['anime_id']
            if a_id in user_watched_item.get(user_id,[]):
                continue
            else:
                rec.append(a_id)
            if len(rec) >= K:
                break
        return rec


# 为测试集里的全部用户进行召回
def get_test_gender_age_recall(test, gender_ranking, user_watched_item, K=20):
    testing = pd.DataFrame({'user_id':test.user_id.unique()})
    testing['anime_recall'] = testing['user_id'].apply(user_gender_age_recall, gender_age_ranking=gender_age_ranking, user_watched_item=user_watched_item, K=K)
    return testing


test = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test.pkl') 
user_watched_item = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\user_watched_item.pkl')
gender_age_ranking = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\gender_age_ranking.pkl')


testing = pd.DataFrame(test.user_id.unique())
testing.columns = ['user_id']
# =============================================================================
# start = time.time()
# res = []
# for index, uid in testing.itertuples(name=False):
#     if (index+1) % 100 == 1:
#         print('epoch:', index+1, '/', 51078)
#     recall_list = user_gender_age_recall(uid, gender_age_ranking, user_watched_item)
#     res.append(recall_list)
# 
# end = time.time()
# print('time used:', (end-start)/60, 'minutes') 
# =============================================================================


start = time.time()
#tqdm.pandas(desc='pandas bar')
testing['anime_recall'] = testing['user_id'].apply(user_gender_age_recall, gender_age_ranking=gender_age_ranking, user_watched_item=user_watched_item)
end = time.time()
print('time used:', (end-start)/60, 'minutes') 




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


test_gender_age_recall = explode(df = testing, group_by_field = 'user_id', explode_field = 'anime_recall')
test_gender_age_recall['anime_recall'] = test_gender_age_recall['anime_recall'].astype(int)
test_gender_age_recall['recall_channel'] = 'gender+age'   # 添加一列召回渠道
test_gender_age_recall.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test_gender_age_recall.pkl')

