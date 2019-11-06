# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:39:57 2019
给csv文件创建表头
@author: tqd95
"""

import pandas as pd
train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_non_zero.pkl')  # 包含1236的非零分以及4
train_non_zero = train_non_zero.head(20)
df = pd.DataFrame(columns = train_non_zero.columns)
df.to_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.csv')

train_zero=pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_zero.pkl')  # 包含123的零分
train_zero = train_zero.head(20)
df = pd.DataFrame(columns = train_zero.columns)
df.to_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_zero.csv')

train_plan_to_watch= pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_plan_to_watch.pkl')  # 包含6的0分
train_plan_to_watch = train_plan_to_watch.head(20)
df = pd.DataFrame(columns = train_plan_to_watch.columns)
df.to_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_plan_to_watch.csv')

test = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test.pkl')
test = test.head(20)
df = pd.DataFrame(columns = test.columns)
df.to_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test.csv')




train_non_zero.to_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.csv',
                      header=False,mode='a')

t = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.csv')
t = t.set_index('Unnamed: 0')
t.index.name = ''



train_zero=pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_zero.pkl')  # 包含123的零分
train_zero = train_zero.head(20)

train_plan_to_watch= pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_plan_to_watch.pkl')  # 包含6的0分
train_plan_to_watch = train_plan_to_watch.head(20)

test = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test.pkl')
test = test.head(20)


df = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.csv')
df.dropna().to_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.csv')


df2 = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.csv')
train_non_zero.to_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.csv',
                      header=False,mode='a')