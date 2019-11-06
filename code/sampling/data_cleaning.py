# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:59:43 2019
MyAnimeList数据集清洗
@author: tqd95
"""

import numpy as np
import pandas as pd
import re
# 原始数据
AnimeList = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\AnimeList.csv')
UserList = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\UserList.csv')
UserAnimeList_part = pd.read_csv(filepath_or_buffer=r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\UserAnimeList.csv',
                                 nrows=300)
# filter数据
anime_filter = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\anime_filtered.csv')
user_filter = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\users_filtered.csv')
useranimelist_filter_part = pd.read_csv(filepath_or_buffer=r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\animelists_filtered.csv',
                                        nrows=300)

# filter的clean数据
anime_clean = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\anime_cleaned.csv')
user_clean = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\users_cleaned.csv')
useranimelist_clean_part = pd.read_csv(filepath_or_buffer=r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\animelists_cleaned.csv',
                                        nrows=3000000)



## 以用户信息为例，观察占用内存情况
useranimelist_clean_part.info(memory_usage='deep')

## 看看不同类型数据的内存占用情况：
for dtype in ('float','int','object'):
    df = useranimelist_clean_part.select_dtypes(include=[dtype])
    avg_use_b = df.memory_usage(deep=True).mean()
    avg_use_mb = avg_use_b/(1024**2)
    sum_use_b = df.memory_usage(deep=True).sum()
    sum_use_mb = sum_use_b/(1024**2)
    print('%s类型的列平均占用内存为%5.2f MB' % (dtype,avg_use_mb))
    print('%s类型的列总占用内存为%5.2f MB' % (dtype,avg_use_mb))
    print('=============================')
    
## 用子类型优化整型列
int_columns = useranimelist_clean_part.select_dtypes(include=['int64']).columns
useranimelist_clean_part[int_columns] = useranimelist_clean_part.select_dtypes(include=['int64']).apply(pd.to_numeric,downcast='unsigned')

useranimelist_clean_part.drop('my_tags',axis = 1, inplace=True)
useranimelist_clean_part.drop('my_start_date',axis = 1, inplace=True)
useranimelist_clean_part.drop('my_finish_date',axis = 1, inplace=True)

## 对类别唯一值少于50%的object列，向下转化为category类型
useranimelist_clean_part['last_update_year'] = useranimelist_clean_part['my_last_updated'].apply(lambda x: int(re.match('^(\d+)-(\d+)-(\d+).*$',x).group(1)))
useranimelist_clean_part['last_update_month'] = useranimelist_clean_part['my_last_updated'].apply(lambda x: int(re.match('^(\d+)-(\d+)-(\d+).*$',x).group(2)))
useranimelist_clean_part['last_update_day'] = useranimelist_clean_part['my_last_updated'].apply(lambda x: int(re.match('^(\d+)-(\d+)-(\d+).*$',x).group(3)))
useranimelist_clean_part.drop('my_last_updated',axis = 1, inplace=True)

int_columns = useranimelist_clean_part.select_dtypes(include=['int64']).columns
useranimelist_clean_part[int_columns] = useranimelist_clean_part.select_dtypes(include=['int64']).apply(pd.to_numeric,downcast='unsigned')

## 把用户名join上用户id
useranimelist_clean_part = useranimelist_clean_part.merge(user_clean,how='inner',on='username').iloc[:,1:11]
int_columns = useranimelist_clean_part.select_dtypes(include=['int64']).columns
useranimelist_clean_part[int_columns] = useranimelist_clean_part.select_dtypes(include=['int64']).apply(pd.to_numeric,downcast='unsigned')


# 单独看某个用户的记录
df = useranimelist_clean_part.loc[useranimelist_clean_part.user_id==37326,:]
df = df.sort_values(by=['last_update_year','last_update_month','last_update_day'],ascending=[True,True,True])

# 看一下各个status里有无评分的比例：
a = useranimelist_clean_part.groupby('my_status').apply(lambda df:pd.DataFrame(df['my_score'].value_counts().sort_index()))
a = a.unstack()
a = a['my_score']
a.loc[:,'num_of_nonzero'] = a.apply(lambda x:sum(x[1:]), axis = 1)
a['non_zero_percentage'] = a['num_of_nonzero']/(a['num_of_nonzero']+a[0])

b = useranimelist_clean_part['my_status'].value_counts().sort_index()
a['status_percentage'] = b/sum(b)


## 看一下各个status下的打分比例情况：
res = pd.DataFrame(index = [1,2,3,4,5,6])
for i in range(0,5):
    c =     
a.iloc[i,0:11]/sum(a.iloc[i,0:11])
    c = c.apply(lambda x:str(round(100*x,2))+'%')
    res= pd.concat([res,c],axis = 1)

res2 = pd.DataFrame(index = [1,2,3,4,5,6])
for i in range(0,5):
    c = a.iloc[i,1:11]/sum(a.iloc[i,1:11])
    c = c.apply(lambda x:str(round(100*x,2))+'%')
    res2= pd.concat([res2,c],axis = 1)
    
    
    
    
    


