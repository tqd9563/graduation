# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:46:48 2019

@author: tqd95
"""

import numpy as np
import pandas as pd
test = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test.pkl') 
cond = test['start'] > '2017-06-30'
test.loc[cond,'is_new_anime'] = 1
test.loc[-cond,'is_new_anime'] = 0

t = test.groupby('user_id').agg({'is_new_anime':[np.mean,'count']})
t.columns = t.columns.get_level_values(0)
t.columns = ['pct','cnt']
t['new_cnt'] = t['cnt']*t['pct']

res = pd.DataFrame({'pct':t.pct.describe(),
                    'new_cnt':t.new_cnt.describe(),
                    'old_cnt':(t['cnt']*(1-t.pct)).describe(),
                    'total_cnt':t.cnt.describe()})