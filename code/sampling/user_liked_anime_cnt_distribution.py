# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 09:59:01 2019
采样用户喜欢动漫数分布
@author: tqd95
"""
import pandas as pd
import seaborn as sns
from scipy.stats import norm 
train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.pkl')

# 用户喜欢数量的分布情况
user_liked_cnt = train_non_zero.groupby('user_id')['label'].sum()
user_liked_cnt = user_liked_cnt.to_frame().reset_index().rename(columns = {'label':'cnt'})

sns.distplot(user_liked_cnt['cnt'],fit=norm)

# 统计分位数
full_quantile = pd.DataFrame({'q':list(map(lambda x:x/10,range(1,10)))})
cnt = []
for i in range(1,10):
    cnt.append(user_liked_cnt['cnt'].quantile(i/10))
full_quantile['full_liked_cnt']=cnt  


train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_non_zero.pkl')
user_liked_cnt_sampling = train_non_zero.groupby('user_id')['label'].sum()
user_liked_cnt_sampling = user_liked_cnt_sampling.to_frame().reset_index().rename(columns = {'label':'cnt'})

cnt = []
for i in range(1,10):
    cnt.append(user_liked_cnt_sampling['cnt'].quantile(i/10))
full_quantile['sampling_liked_cnt']=cnt  





# 比较double sampling的分位数

seeds = [1002,1005,1007,1008,1010]
for seed in seeds:
    train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_non_zero.pkl')
    user_clean = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\sampling_user.pkl')
    user_clean = user_clean.sample(frac = 0.05, random_state = seed)
    train_non_zero = pd.merge(train_non_zero, user_clean.loc[:,['user_id']], on='user_id')
    user_liked_cnt_sampling = train_non_zero.groupby('user_id')['label'].sum()
    user_liked_cnt_sampling = user_liked_cnt_sampling.to_frame().reset_index().rename(columns = {'label':'cnt'})

    cnt = []
    for i in range(1,10):
        cnt.append(user_liked_cnt_sampling['cnt'].quantile(i/10))
    col_name = 'double_sampling_' + str(seed)
    full_quantile[col_name]=cnt 



seeds = [1001,1002,1003,1004,1005]
for seed in seeds:
    col_name = 'double_sampling_' + str(seed)
    res = ((full_quantile['sampling_liked_cnt'] - full_quantile[col_name])**2).sum()
    print('seed=%s的残差为:%.2f' % (seed,res))



seeds = [1001,1002,1003,1004,1005]
for seed in seeds:
    train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_non_zero.pkl')
    user_clean = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\sampling_user.pkl')
    user_clean = user_clean.sample(frac = 0.05, random_state = seed)
    train_non_zero = pd.merge(train_non_zero, user_clean.loc[:,['user_id']], on='user_id')
    user_liked_cnt_sampling = train_non_zero.groupby('user_id')['label'].sum()
    user_liked_cnt_sampling = user_liked_cnt_sampling.to_frame().reset_index().rename(columns = {'label':'cnt'})
    path = r'C:\Users\tqd95\Desktop\user_liked_cnt_' + str(seed) + '.csv'
    user_liked_cnt_sampling.to_csv(path,index=False)

user_liked_cnt.to_csv(r'C:\Users\tqd95\Desktop\user_liked_cnt.csv',index=False)
user_liked_cnt_sampling.to_csv(r'C:\Users\tqd95\Desktop\user_liked_cnt_sampling.csv',index=False)








train_non_zero.query('my_status==1')['my_score'].describe()

train_non_zero.query('my_status==2')['my_score'].describe()














