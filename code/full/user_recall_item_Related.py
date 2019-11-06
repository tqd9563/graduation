# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 18:59:14 2019

@author: tqd95
"""

import time
import pandas as pd
from tqdm import tqdm_notebook, _tqdm_notebook, tqdm
from pandarallel import pandarallel

train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_non_zero.pkl')
anime_clean = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\anime_cleaned.csv')

def explode(df, group_by_field, explode_field):
    '''
    @group_field: 分组键, string形式
    @explode_field: 根据df的explode_field列拆分成多行, string形式
    '''
    df1 = df[explode_field].apply(pd.Series).stack().rename(explode_field)
    df2 = df1.to_frame().reset_index(1, drop=True)
    res = df2.join(df[group_by_field]).reset_index(drop=True)
    return res.loc[:, [group_by_field, explode_field]]

def get_related(x, related_type):
    if isinstance(eval(x), list):
        return [{}]
    elif not eval(x).get(related_type):
        return [{}]
    else:
        return eval(x).get(related_type)
    
def get_anime_related_recall():
    anime_clean = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\anime_cleaned.csv')
    a = anime_clean.loc[:, ['anime_id', 'title_japanese', 'related']]
    a['Sequel_anime_id'] = a['related'].apply(get_related, related_type='Sequel') \
                                    .apply(lambda x: [dic.get('mal_id') for dic in x])
    a['Prequel_anime_id'] = a['related'].apply(get_related, related_type='Prequel') \
                                    .apply(lambda x: [dic.get('mal_id') for dic in x])
    a['Parent_story_anime_id'] = a['related'].apply(get_related, related_type='Parent story') \
                                    .apply(lambda x: [dic.get('mal_id') for dic in x])
    a['Side_story_anime_id'] = a['related'].apply(get_related, related_type='Side story') \
                                    .apply(lambda x: [dic.get('mal_id') for dic in x])

    Sequel_recall = explode(a, 'anime_id', 'Sequel_anime_id').rename(columns={'Sequel_anime_id': 'recall'})
    Sequel_recall['Related_type'] = 'Sequel'

    Prequel_recall = explode(a, 'anime_id', 'Prequel_anime_id').rename(columns={'Prequel_anime_id': 'recall'})
    Prequel_recall['Related_type'] = 'Prequel'

    Parent_story_recall = explode(a, 'anime_id', 'Parent_story_anime_id').rename(columns={'Parent_story_anime_id': 'recall'})
    Parent_story_recall['Related_type'] = 'Parent_story'

    Side_story_recall = explode(a, 'anime_id', 'Side_story_anime_id').rename(columns={'Side_story_anime_id': 'recall'})
    Side_story_recall['Related_type'] = 'Side_story'

    Related_recall = pd.concat([Sequel_recall, Prequel_recall, Parent_story_recall, Side_story_recall], axis=0)
    Related_recall = pd.merge(anime_clean.loc[:, 'anime_id'].to_frame(), Related_recall, on='anime_id', how='left')
    Related_recall.fillna({'recall': 0}, inplace=True)
    Related_recall['recall'] = Related_recall['recall'].astype(int)

    return Related_recall

related_recall = get_anime_related_recall()


# 获取每个用户在训练集里喜欢的每部动漫的打分,倒序
# 类似user_item,只不过这次不是set而是一个dict
def get_user_item_ranking(train):
    '''
    @train: 非零打分训练集
    统计的是用户在训练集里喜欢的动漫的打分排序
    输出的是一个series,其中index就是user_id, value是一个dict,形如:{anime_id: my_score}
    '''
    res = train.loc[train.label == 1, ['user_id', 'anime_id', 'my_score']] \
                .groupby('user_id') \
                .apply(lambda x: x.sort_values(by='my_score', ascending=False)) \
                .loc[:,['anime_id','my_score']] \
                .reset_index() \
                .drop('level_1', axis=1)
    res.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\user_item_ranking.pkl')        
    return res

start = time.time()     
user_item_ranking = get_user_item_ranking(train_non_zero)       # 0.78min
end = time.time()
print('time used:', (end-start)/60, 'minutes') 


# 得到每个用户看过的所有动漫
def get_user_watched_item(train):
    ## 返回一个series, index是user_id, value是用户看过的所有动漫(包括低分)
    #user_stats = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\user_stats.pkl')
    #user_stats = user_stats.loc[:, ['user_id']]
    user_watched_item = train.groupby('user_id').agg(lambda x: set(x.anime_id))
    user_watched_item = user_watched_item.iloc[:, 0]
    user_watched_item = pd.DataFrame(user_watched_item)
    user_watched_item = user_watched_item.reset_index()
    user_watched_item['num'] = user_watched_item['anime_id'].apply(
        lambda x: len(x))
    #res = pd.merge(user_stats, user_watched_item, on = 'user_id', how = 'left')
    # res.anime_id.fillna('',inplace=True)
    res = user_watched_item.iloc[:, :2].set_index('user_id')['anime_id']
    res.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\user_watched_item.pkl')
    return res


start = time.time()     
user_watched_item = get_user_watched_item(train_non_zero)   # 1.7min
end = time.time()
print('time used:', (end-start)/60, 'minutes') 


# 测试集单个用户的related召回
def user_related_recall(user_id, anime_stats, user_item_ranking, related_recall, user_watched_item, K1=10, K2=20):
    '''
    @user_id: 用户id
    @user_item_ranking: 用户喜欢的动漫按打分高低排序
    @related_recall: 给每一部动漫按照related字段联系到别的动漫
    @user_watched_item: 每个用户看过的动漫,最后要做过滤
    '''
    # 如果user_id在user_item_ranking里找不到,则返回空
    if not user_id in set(user_item_ranking.user_id):
        return []
    #user_liked_anime_list = user_item_ranking[user_id]
    user_liked_anime_list = user_item_ranking.loc[user_item_ranking.user_id==user_id,'anime_id']
    total_anime = set(anime_stats.a_anime_id)
    res = []
    cnt = 0
    for index,a_id in user_liked_anime_list.iteritems():
        related = related_recall.loc[related_recall.anime_id == a_id, :]
        for index, row in related.iterrows():
            recall = row['recall']
            #recall_type = row['Related_type']
            # 这个判断条件比较复杂,需要有四条:
            # recall非空, 没被用户看过, 还未被其他动漫通过related的方式召回(防止多个动漫related同一个动漫)
            # 以及recall的动漫在anime_stats里
            cond1 = recall and recall not in user_watched_item[user_id]
            cond2 = recall not in res and recall in total_anime
            if cond1 and cond2:
                cnt += 1
                res.append(recall)
                #res =pd.concat([res,pd.DataFrame({'user_id':user_id,'anime_recall':recall,'recall_type':recall_type},index=[cnt])],axis=0)
            if cnt >= K1:
                break
        if len(res) >= K2:  # 只要找到了K2部related 动漫就停止
            break
    return res



# =============================================================================
# user_item_ranking = user_item_ranking.reset_index() \
#                         .drop('',axis=1) \
#                         .set_index('user_id')['anime_id']
# =============================================================================
                        

def get_test_related_recall(test, anime_stats, user_item_ranking, related_recall, user_watched_item, K1=10, K2=10):
    '''
    @test: 不加任何处理的测试集
    @user_item_ranking：用户喜欢的动漫按打分高低排序
    @related_recall: 给每一部动漫按照related字段联系到别的动漫
    @K：根据related关系,给每个用户最多召回的动漫数,默认10部
    '''
    testing = pd.DataFrame(test.user_id.unique())
    testing.columns = ['user_id']
    testing['anime_recall'] = testing['user_id'].apply(user_related_recall, anime_stats=anime_stats, user_item_ranking=user_item_ranking, 
                                                       related_recall=related_recall, user_watched_item=user_watched_item, K1=K1, K2=K2)
    #testing.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test_related_recall_10.pkl')
    return testing



# 下面的代码估计要跑1h40min
test = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test.csv')
user_item_ranking = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\user_item_ranking.pkl')
user_watched_item = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\user_watched_item.pkl')
anime_stats = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\anime_stats.pkl')
related_recall = get_anime_related_recall()


test_related_recall = get_test_related_recall(test, anime_stats, user_item_ranking, related_recall, 
                                              user_watched_item, K1=10, K2=10)

# 把召回的动画id拆分成多行(这样一个user_id就会占多行)
test_related_recall = explode(df=test_related_recall, group_by_field='user_id', explode_field='anime_recall')
test_related_recall['recall_channel'] = 'related'   # 添加一列召回渠道
test_related_recall['anime_recall'] = test_related_recall['anime_recall'].astype(int)

test_related_recall.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test_related_recall_10.pkl')


