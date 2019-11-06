# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 18:14:05 2019
探索数据
@author: tqd95
"""


# 看一下各个status里有无评分的比例：
a = useranimelist_clean.groupby('my_status').apply(lambda df:pd.DataFrame(df['my_score'].value_counts().sort_index()))
a = a.unstack()
a = a['my_score']
a['non_zero_percentage']  = (a.sum(axis = 1,skipna=True)-a[0])/a.sum(axis = 1,skipna=True)
a['non_zero_percentage'] = a['non_zero_percentage'].apply(lambda x:round(x,3))

b = useranimelist_clean['my_status'].value_counts().sort_index()
a['status_percentage'] = round(b/sum(b),3)
a = a.iloc[[1,2,3,4,6],:]


## 看一下各个status下的打分比例情况：
res = pd.DataFrame(index = [1,2,3,4,6])
for i in range(0,5):
    c = a.iloc[i,0:11]/sum(a.iloc[i,0:11])
    c = c.apply(lambda x:round(x,3))
    res= pd.concat([res,c],axis = 1)
    
    
res2 = pd.DataFrame(index = [1,2,3,4,5,6])
for i in range(0,5):
    c = a.iloc[i,1:11]/sum(a.iloc[i,1:11])
    c = c.apply(lambda x:round(x,3))
    res2= pd.concat([res2,c],axis = 1)
## 看各个status有打分的样本中的平均得分(除去那些非零的打分外)
m = useranimelist_clean.loc[useranimelist_clean.my_score!=0,['my_score','my_status']].groupby('my_status')
m = pd.DataFrame(m.mean())
m = m.iloc[[1,2,3,4,6],:].T


## 看下时间

useranimelist_clean['p_date'] = useranimelist_clean['p_date'].apply(pd.to_numeric,downcast='unsigned')

## 统计日期的各分位数：(结果显示日期的90%分位数是2017-07-21)
date_quantile = pd.DataFrame({'q':list(map(lambda x:x/10,range(1,10)))})
date = []
for i in range(1,10):
    date.append(useranimelist_clean['p_date'].quantile(i/10))
date_quantile['date']=date  