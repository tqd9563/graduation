# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:30:17 2019
给定anime_id，判断是新番还是旧番
@author: tqd95
"""

import pandas as pd

anime_stats = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\anime_stats.pkl')
a = anime_stats.loc[:,['a_anime_id','a_start']]
del anime_stats
cond = a['a_start'] > '2017-06-30'
a.loc[cond,'is_new_anime'] = 1
a.loc[-cond,'is_new_anime'] = 0
del a['a_start']
a['is_new_anime'] = a['is_new_anime'].astype(int)
new_anime_mapping = a.set_index('a_anime_id')['is_new_anime'].to_dict()
del a