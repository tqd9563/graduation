# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:00:21 2019

@author: tqd95
"""

import pandas as pd

test_recall_mapping = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\test_recall_channel_mapping.pkl')
train_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\train_zero.pkl') # 包含123的零分
train_plan_to_watch = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\train_plan_to_watch.pkl') # 包含6的0分


# 检查我最后给每个用户召回的动漫，有多少是这些用户曾经打了0分的
train_zero = train_zero.loc[:,['user_id','anime_id']]
train_plan_to_watch = train_plan_to_watch.loc[:,['user_id','anime_id']]
recall = test_recall_mapping.loc[:,['user_id','anime_recall']]
t = pd.merge(train_zero,recall, left_on=['user_id','anime_id'], right_on=['user_id','anime_recall'],how='outer')
t2 = pd.merge(train_plan_to_watch,recall, left_on=['user_id','anime_id'], right_on=['user_id','anime_recall'],how='outer')
result = t.groupby('user_id').apply(lambda df:len(df.dropna()))
result2 = t2.groupby('user_id').apply(lambda df:len(df.dropna()))































