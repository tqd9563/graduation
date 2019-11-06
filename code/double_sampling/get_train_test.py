# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:24:12 2019
10w全量数据下的采样数据
@author: tqd95
"""

import pandas as pd

# 5%用户信息(5436个用户)
user_clean = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\sampling_user.pkl')  # 采样用户集, 比例5%
train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_non_zero.pkl')  # 包含1236的非零分以及4
train_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_zero.pkl')  # 包含123的零分
train_plan_to_watch = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_plan_to_watch.pkl')  # 包含6的0分
test = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test.pkl')

# 对5436名用户（对应训练集大小为89w行）随机采样其中5%
user_clean = user_clean.sample(frac=0.05, random_state=1001)  # 随机采样5%比例的用户，种子为1001
train_non_zero = pd.merge(train_non_zero, user_clean.loc[:,['user_id']], on = 'user_id')
train_zero = pd.merge(train_zero, user_clean.loc[:,['user_id']], on = 'user_id')
train_plan_to_watch = pd.merge(train_plan_to_watch, user_clean.loc[:,['user_id']], on = 'user_id')
test = pd.merge(test, user_clean.loc[:,['user_id']], on = 'user_id')


# 保存到本地
train_non_zero.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\double_sampling\train_non_zero.pkl')  # 包含1236的非零分以及4
train_zero.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\double_sampling\train_zero.pkl')  # 包含123的零分
train_plan_to_watch.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\double_sampling\train_plan_to_watch.pkl')  # 包含6的0分
test.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\double_sampling\test.pkl')




