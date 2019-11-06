# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 20:11:36 2019
用户冷启动推荐
@author: tqd95
"""
import numpy as np
import pandas as pd
import time


train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_non_zero.pkl') # 包含1236的非零分以及4

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
    user_stats = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\user_stats.pkl')
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
gender_age_ranking = pd.DataFrame()
for gender in genders:
    for age in ages:
        gender_age_ranking = pd.concat([gender_age_ranking, get_gender_age_ranking(train_non_zero,gender,age)],axis=0)
        

# 为单个测试集用户进行召回
def user_gender_age_recall(user_id, gender_age_ranking, user_watched_item, K=20):
    '''
    @gender_age_ranking: 根据性别以及年龄分段的string,找到对应分段的top动漫,然后加以过滤后召回
    '''
    user_stats = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\user_stats.pkl')
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

def get_user_watched_item(train):
    ## 返回一个series, index是user_id, value是用户看过的所有动漫(包括低分)
    user_stats = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\user_stats.pkl')
    user_stats = user_stats.loc[:,['user_id']]
    user_watched_item = train.groupby('user_id').agg(lambda x: set(x.anime_id))
    user_watched_item = user_watched_item.iloc[:,0]
    user_watched_item = pd.DataFrame(user_watched_item)
    user_watched_item = user_watched_item.reset_index()
    user_watched_item['num'] = user_watched_item['anime_id'].apply(lambda x:len(x))
    #res = pd.merge(user_stats, user_watched_item, on = 'user_id', how = 'left')
    #res.anime_id.fillna('',inplace=True)
    return user_watched_item.iloc[:,:2].set_index('user_id')['anime_id']

user_watched_item = get_user_watched_item(train_non_zero)
user_watched_item = dict(user_watched_item)

test = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test.pkl') 
start = time.time()
test_gender_age_recall = get_test_gender_age_recall(test, gender_age_ranking, user_watched_item)
end = time.time()
print('time used:', end-start) # 用时38.5s


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


test_gender_age_recall = explode(df = test_gender_age_recall, group_by_field = 'user_id', explode_field = 'anime_recall')
test_gender_age_recall['anime_recall'] = test_gender_age_recall['anime_recall'].astype(int)
test_gender_age_recall['recall_channel'] = 'gender+age'   # 添加一列召回渠道
test_gender_age_recall.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test_gender_age_recall.pkl')



# =============================================================================
# 
# test_age_recall = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\test_age_recall.pkl')
# 
# 
# gender_age_ranking.groupby(['gender','age']).size()
# 
# test_gender_age_recall['len'] = test_gender_age_recall.anime_recall.apply(lambda x:len(x))
# test_gender_age_recall.anime_recall.apply(lambda x:len(x)).value_counts()
# test_gender_age_recall.len.describe()
# 
# t_union = pd.concat([test_gender_recall,test_age_recall],axis = 0).drop_duplicates(subset=['user_id','anime_recall'])
# t_union = t_union.groupby('user_id').apply(lambda df:list(df.anime_recall)).reset_index()
# t_union = t_union.rename(columns = {0:'anime_recall'})
# t_union['len'] = t_union.anime_recall.apply(lambda x:len(x))
# =============================================================================