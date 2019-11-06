# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:41:50 2019

@author: tqd95
"""

import pandas as pd
import re
import time
import warnings


def preprocess(df):
    df = pd.merge(df, user_clean, on='username', how='inner')
    df = df.loc[:, ['user_id', 'anime_id', 'my_watched_episodes', 'my_score', 'my_status',
                    'last_update_year', 'last_update_month', 'last_update_day']]
    df['last_update_month'] = df['last_update_month'].apply(
        lambda x: str(x) if len(str(x)) == 2 else '0' + str(x))
    df['last_update_day'] = df['last_update_day'].apply(
        lambda x: str(x) if len(str(x)) == 2 else '0' + str(x))
    df['last_update_date'] = df['last_update_year'].astype(
        str) + '-' + df['last_update_month'] + '-' + df['last_update_day']
    del df['last_update_year']
    del df['last_update_month']
    del df['last_update_day']
    return df

def train_test_split():
    '''
    @df: 分块读取的useranimelist数据集,chunkSize=100w
    返回结果:
    @train_non_zero: 最终用于模型训练的训练集,包含status=1,2,3,6的非零打分样本和statue=4的所有样本
    @train_zero：包含status=1,2,3的零分样本
    @train_plan_to_watch：包含status=6的零分样本
    @test: 不加任何处理的最原始的测试集
    '''
    
    # 一次读入100w行数据，大约需要3min
    chunkSize = 1000000
    reader = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\animelists_cleaned.csv',
                         chunksize=chunkSize,
                         usecols=['username', 'anime_id', 'my_watched_episodes', 'my_score',
                                  'my_status', 'my_last_updated'])
    i = 1
    for df in reader:
        start = time.time()
        print('epoch:', i, '/', 32, '...')
        df['last_update_year'] = df['my_last_updated'].apply(
            lambda x: int(re.match('^(\d+)-(\d+)-(\d+).*$', x).group(1)))
        df['last_update_month'] = df['my_last_updated'].apply(
            lambda x: int(re.match('^(\d+)-(\d+)-(\d+).*$', x).group(2)))
        df['last_update_day'] = df['my_last_updated'].apply(
            lambda x: int(re.match('^(\d+)-(\d+)-(\d+).*$', x).group(3)))
        df.drop('my_last_updated', axis=1, inplace=True)
        
        df = preprocess(df)

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
        train_non_zero.loc[:, ['my_status', 'label']] = train_non_zero.loc[:, [
            'my_status', 'label']].apply(pd.to_numeric, downcast='unsigned')
        # 5. test的标记相对容易，status=123并且看了的就令label = 1，dropped和plan的还有没看的label = 0
        
        # 分块处理完毕后，追加写入到本地csv中去
        train_non_zero.to_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.csv',
                              header = False, mode = 'a')
        del train_non_zero
        train_zero.to_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_zero.csv',
                          header = False, mode = 'a')
        del train_zero
        train_plan_to_watch.to_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_plan_to_watch.csv',
                                   header = False, mode = 'a')
        del train_plan_to_watch
        test.to_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test.csv',
                    header = False, mode = 'a')
        del test
        
        end = time.time()
        print('time used:', (end-start)/60, 'minutes')
        i += 1


if __name__ == '__main__':
    user_clean = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\users_cleaned.csv')
    anime_clean = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\anime_cleaned.csv')
    anime_clean['start'] = anime_clean['aired'].apply(lambda x: eval(x)['from'])
    anime_clean['end'] = anime_clean['aired'].apply(lambda x: eval(x)['to'])
    warnings.filterwarnings("ignore")
    beg = time.time()
    train_test_split()
    ending = time.time()
    print('total time used:', (ending-beg)/60, 'minutes')
