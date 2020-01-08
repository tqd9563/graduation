# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:23:24 2019

@author: tqd95
"""
import pandas as pd
import re

train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_non_zero.pkl')  # 包含1236的非零分以及4
train_zero= pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_zero.pkl')  # 包含123的零分
train_plan_to_watch= pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_plan_to_watch.pkl')  # 包含6的0分
test= pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test.pkl')

user_anime_list = pd.concat([train_non_zero,train_zero,train_plan_to_watch,test],axis=0)

user_anime_list['p_year'] = user_anime_list['last_update_date'].apply(lambda x: int(re.match('^(\d+)-(\d+)-(\d+).*$', x).group(1)))
user_anime_list['p_month'] = user_anime_list['last_update_date'].apply(lambda x: int(re.match('^(\d+)-(\d+)-(\d+).*$', x).group(2)))
user_anime_list['p_day'] = user_anime_list['last_update_date'].apply(lambda x: int(re.match('^(\d+)-(\d+)-(\d+).*$', x).group(3)))

user_anime_list['p_month'] = user_anime_list['p_month'].apply(lambda x: str(x) if len(str(x)) == 2 else '0' + str(x))
user_anime_list['p_day'] = user_anime_list['p_day'].apply(lambda x: str(x) if len(str(x)) == 2 else '0' + str(x))
user_anime_list['p_year'] = user_anime_list['p_year'].astype(str)

user_anime_list['p_date'] = user_anime_list.p_year + user_anime_list.p_month + user_anime_list.p_day
user_anime_list['p_date'] = user_anime_list['p_date'].astype(int)

date_quantile = pd.DataFrame({'q':list(map(lambda x:x/10,range(1,10)))})
date = []
for i in range(1,10):
    date.append(user_anime_list['p_date'].quantile(i/10))
date_quantile['date']=date 