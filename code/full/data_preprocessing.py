# -*- coding: utf-8 -*-
# @Author: tqd95
# @Date:   2019-10-10 11:39:08
# @Last Modified by:   tqd95
# @Last Modified time: 2019-10-10 11:51:02

import numpy as np
import pandas as pd
import re
import time

# 逐块读取数据，细微整理


def preprocess(df):
    user_clean = pd.read_csv(
        r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\users_cleaned.csv')
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


def read_useranime():
    # 一次读入100w行数据，大约需要3min
    chunkSize = 1000000
    reader = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\animelists_cleaned.csv',
                         chunksize=chunkSize,
                         usecols=['username', 'anime_id', 'my_watched_episodes', 'my_score',
                                  'my_status', 'my_last_updated'])
    anime_clean = pd.read_csv(
        r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\anime_cleaned.csv')
    df = pd.DataFrame()
    for data in reader:
        data['last_update_year'] = data['my_last_updated'].apply(
            lambda x: int(re.match('^(\d+)-(\d+)-(\d+).*$', x).group(1)))
        data['last_update_month'] = data['my_last_updated'].apply(
            lambda x: int(re.match('^(\d+)-(\d+)-(\d+).*$', x).group(2)))
        data['last_update_day'] = data['my_last_updated'].apply(
            lambda x: int(re.match('^(\d+)-(\d+)-(\d+).*$', x).group(3)))
        data.drop('my_last_updated', axis=1, inplace=True)
        data = preprocess(data)
        int_columns = data.select_dtypes(include=['int64']).columns
        data[int_columns] = data.select_dtypes(
            include=['int64']).apply(pd.to_numeric, downcast='unsigned')
        df = pd.concat([df, data], axis=0, sort=False)
    return df


if __name__ == '__main__':
    user_clean = pd.read_csv(
        r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\users_cleaned.csv')
    anime_clean = pd.read_csv(
        r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\anime_cleaned.csv')
    start = time.time()
    print('start reading')
    print('......')
    useranimelist_clean = read_useranime()
    useranimelist_clean.to_pickle(
        r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\df.pkl')
    print('reading completed.')
    end = time.time()
    print('time used:', (end - start) / 60, 'minutes')
