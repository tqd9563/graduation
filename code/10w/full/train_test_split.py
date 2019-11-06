# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:48:27 2019
10w全量数据
@author: tqd95
"""

import pandas as pd
import time 

# 对全量的10w名用户进行随机抽样，使得最后生成的训练集样本量大约为10w行。
# 这里选择的比例是0.006，大约652名用户的行为数据
user_clean = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\users_cleaned.csv')
user_clean = user_clean.sample(frac = 0.006, random_state=1001)

anime_clean = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\anime_cleaned.csv')
anime_clean['start'] = anime_clean['aired'].apply(lambda x: eval(x)['from'])
anime_clean['end'] = anime_clean['aired'].apply(lambda x: eval(x)['to'])
useranimelist_clean =  pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\df.pkl')

# 生成10w行的非零训练集
def train_test_split(useranimelist_clean, user_clean, anime_clean):
    df = pd.merge(useranimelist_clean, user_clean.loc[:,['user_id']],  on = 'user_id')
    df = pd.merge(df, anime_clean, on='anime_id')
    df = df.loc[:, ['user_id', 'anime_id', 'my_watched_episodes', 'episodes', 'my_score',
                        'my_status', 'last_update_date', 'title', 'score', 'rank', 'start', 'end']]
    int_columns = df.select_dtypes(include=['int64']).columns
    df[int_columns] = df.select_dtypes(include=['int64']).apply(pd.to_numeric, downcast='unsigned')
    
    # 1. 以时间=2017/6/30为界，并删除p_date = 19700101的，以及my_status特殊的记录，同时删除last_update_time早于动画发布日期的记录
    df = df.query('last_update_date != "1970-01-01"')
    df = df.query('my_status == 1 or my_status == 2 or my_status == 3 or my_status == 4 or my_status == 6')

    cond = df['last_update_date'] < df['start']
    df = df.loc[-cond, :]
    train = df.query('last_update_date <= "2017-06-30"')
    # 这里的test没有做任何加工处理
    test = df.query('last_update_date > "2017-06-30"')
    # 2. 修正status的异常：如果看的集数=动漫集数>0,则status = 2
    train.loc[((train.my_status == 1) & (train.my_watched_episodes ==
                                         train.episodes) & (train.episodes > 0)), 'my_status'] = 2
    train.loc[((train.my_status == 3) & (train.my_watched_episodes ==
                                         train.episodes) & (train.episodes > 0)), 'my_status'] = 2
    train.loc[((train.my_status == 4) & (train.my_watched_episodes ==
                                         train.episodes) & (train.episodes > 0)), 'my_status'] = 2
    train.loc[((train.my_status == 6) & (train.my_watched_episodes ==
                                         train.episodes) & (train.episodes > 0)), 'my_status'] = 2
    train.loc[((train.my_status == 6) & (train.my_watched_episodes > 0)), 'my_status'] = 1
    # 3. 打分高于8分的label=1，打分低于8分(非零)的label=0
    train.loc[train.my_score >= 8, 'label'] = 1
    train.loc[((train.my_score < 8) & (train.my_score > 0)), 'label'] = 0
    # 4. 如果状态是dropped, 则除去那些打分高于8分的以外, 即使没有打分信息, 也认为label = 0
    train.loc[((train.my_status == 4) & (train.my_score < 8)), 'label'] = 0
    # 5. 把状态是plan to watch的, 并且没有打分信息的样本单独提取出来，后续加工成用户画像特征
    cond = ((train['my_status'] == 6) & (train['my_score'] == 0))
    train_plan_to_watch = train.loc[cond, :]
    train = train.loc[-cond, :]
    # 6. 现在还剩下的没有标记y的数据，全都是来自status=1,2,3的0分样本
    # 这部分样本暂时没想到什么解决的方法，先转存出来吧
    cond1 = (train['my_status'] == 1) & (train['my_score'] == 0)
    cond2 = (train['my_status'] == 2) & (train['my_score'] == 0)
    cond3 = (train['my_status'] == 3) & (train['my_score'] == 0)
    cond_total = (cond1 | cond2) | cond3
    # 这个train_zero里的样本都是0分样本,来自状态1,2,3
    train_zero = train.loc[cond_total, :]
    # 这个train_non_zero里面没有0分样本(除非是dropped)
    train_non_zero = train.loc[-cond_total, :]

    train_non_zero['label'] = train_non_zero['label'].apply(int)
    del train_plan_to_watch['label']
    del train_zero['label']
    train_non_zero.loc[:, ['my_status', 'label']] = train_non_zero.loc[:, ['my_status', 'label']].apply(pd.to_numeric, downcast='unsigned')
    
    # 处理完毕后，写入本地文件
    train_non_zero.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\10w\full\train_non_zero.pkl')
    train_non_zero.to_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\10w\full\train_non_zero.csv')
    train_zero.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\10w\full\train_zero.pkl')
    train_plan_to_watch.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\10w\full\train_plan_to_watch.pkl')
    test.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\10w\full\test.pkl')
    return 


start = time.time()
train_test_split(useranimelist_clean, user_clean, anime_clean)
end = time.time()
print('time used:', (end-start)/60, 'minutes')



# 看一下训练集和测试集的用户个数情况（抽样的用户一共652人）：
train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\10w\train_non_zero.pkl')
test= pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\10w\test.pkl')
print('train_non_zero:', len(set(train_non_zero.user_id)))
print('test:', len(set(test.user_id)))
print('train_non_zero + test:', len(set(train_non_zero.user_id) | set(test.user_id)))




