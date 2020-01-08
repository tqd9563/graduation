# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:36:16 2019
Related动漫召回
@author: tqd95
"""

import pandas as pd

train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_non_zero.pkl')
anime_clean = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\anime_cleaned.csv')
anime_stats = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\anime_stats.pkl')
a = anime_clean.loc[:, ['anime_id', 'title_japanese', 'related']]

# 每个动漫的related特征都是一个string,通过eval可以解析成一个dict
r0 = eval(a['related'][0])  # 有2个key: 'Adaptation', 'Sequel'
# 有4个key: 'Adaptation', 'Prequel', 'Spin-off', 'Sequel'
r1 = eval(a['related'][7])


# 找出所有可能的key：
a['related_keys'] = a['related'].apply(lambda x: list(
    eval(x).keys()) if not isinstance(eval(x), list) else eval(x))


def explode(df, group_by_field, explode_field):
    '''
    @group_field: 分组键, string形式
    @explode_field: 根据df的explode_field列拆分成多行, string形式
    '''
    df1 = df[explode_field].apply(pd.Series).stack().rename(explode_field)
    df2 = df1.to_frame().reset_index(1, drop=True)
    res = df2.join(df[group_by_field]).reset_index(drop=True)
    return res.loc[:, [group_by_field, explode_field]]


a1 = explode(a, 'anime_id', 'related_keys')
related_keys = a1.groupby('related_keys').size().sort_values(ascending=False)


# 可以认为Adaptation基本上都是漫画,因为我们要推荐的是动漫,不含漫画，所以这部分related可以不考虑了
# 主要的召回目标应该是Sequel, Prequel, Parent story, Side story

def get_related(x, related_type):
    if isinstance(eval(x), list):
        return [{}]
    elif not eval(x).get(related_type):
        return [{}]
    else:
        return eval(x).get(related_type)

# 得到每一部动漫的related召回(其实一共只有3676部动漫是有结果的,其余召回为缺失)
## 如果没有召回结果的话, recall列为0, Related_type列为nan


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


# 选出每个用户评分最高的K部动漫
def get_user_topK_anime(train, K=5):
    u = train.loc[train.label == 1, ['user_id', 'anime_id', 'my_score', 'score']] \
        .groupby('user_id') \
        .apply(lambda x: x.loc[:, ['anime_id', 'my_score', 'score']].sort_values(by='my_score', ascending=False)[:K]) \
        .reset_index()
    u.drop('level_1', axis=1, inplace=True)
    return u


user_topK_anime = get_user_topK_anime(train_non_zero, K=30)


# 获取每个用户在训练集里喜欢的每部动漫的打分,倒序
# 类似user_item,只不过这次不是set而是一个dict
def get_user_item_ranking(train):
    '''
    @train: 非零打分训练集
    输出的是一个series,其中index就是user_id, value是一个dict,形如:{anime_id: my_score}
    '''
    res = train.loc[train.label == 1, ['user_id', 'anime_id', 'my_score', 'score']] \
        .groupby('user_id') \
        .apply(lambda x: x.sort_values(by='my_score', ascending=False)) \
        .reset_index() \
        .set_index('anime_id') \
        .drop('index', axis=1)['my_score'].to_dict()

    return res


user_item_ranking = get_user_item_ranking(train_non_zero)


# 得到每个用户看过的所有动漫
def get_user_watched_item(train):
    ## 返回一个series, index是user_id, value是用户看过的所有动漫(包括低分)
    user_stats = pd.read_pickle(
        r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\user_stats.pkl')
    user_stats = user_stats.loc[:, ['user_id']]
    user_watched_item = train.groupby('user_id').agg(lambda x: set(x.anime_id))
    user_watched_item = user_watched_item.iloc[:, 0]
    user_watched_item = pd.DataFrame(user_watched_item)
    user_watched_item = user_watched_item.reset_index()
    user_watched_item['num'] = user_watched_item['anime_id'].apply(
        lambda x: len(x))
    #res = pd.merge(user_stats, user_watched_item, on = 'user_id', how = 'left')
    # res.anime_id.fillna('',inplace=True)
    return user_watched_item.iloc[:, :2].set_index('user_id')['anime_id']


user_watched_item = get_user_watched_item(train_non_zero)
user_watched_item = dict(user_watched_item)


# 测试集单个用户的related召回
def user_related_recall(user_id, anime_stats, user_item_ranking, related_recall, user_watched_item, K=10):
    '''
    @user_id: 用户id
    @user_item_ranking: 用户喜欢的动漫按打分高低排序
    @related_recall: 给每一部动漫按照related字段联系到别的动漫
    @user_watched_item: 每个用户看过的动漫,最后要做过滤
    '''
    # 如果user_id在user_item_ranking里找不到,则返回空
    if not user_id in user_item_ranking.index:
        return []
    user_liked_anime_list = user_item_ranking[user_id]
    total_anime = set(anime_stats.a_anime_id)
    res = []
    cnt = 0
    for a_id, a_score in user_liked_anime_list.items():
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
            if cnt >= K:
                break
        if cnt >= K:
            break
    return res


def get_test_related_recall(test, anime_stats, user_item_ranking, related_recall, user_watched_item, K=10):
    '''
    @test: 不加任何处理的测试集
    @user_item_ranking：用户喜欢的动漫按打分高低排序
    @related_recall: 给每一部动漫按照related字段联系到别的动漫
    @K：根据related关系,给每个用户最多召回的动漫数,默认10部
    '''
    testing = pd.DataFrame({'user_id': test.user_id.unique()})
    testing['anime_recall'] = testing['user_id'].apply(user_related_recall, anime_stats=anime_stats, user_item_ranking=user_item_ranking, related_recall=related_recall, user_watched_item=user_watched_item, K=K)
    return testing


test = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test.pkl')

test_related_recall = get_test_related_recall(test, anime_stats, user_item_ranking, related_recall, user_watched_item)

# 把召回的动画id拆分成多行(这样一个user_id就会占多行)
test_related_recall = explode(df=test_related_recall, group_by_field='user_id', explode_field='anime_recall')
test_related_recall['recall_channel'] = 'related'   # 添加一列召回渠道
test_related_recall.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test_related_recall.pkl')

# =============================================================================
# 
# # 这个方法不好，因为还要和训练集里喜欢的动漫作过滤。。
# # 用户related召回
# def get_user_related_recall(train, K1=30, K2=10):
#     '''
#     @K1: 挑选每个用户在训练集里最喜欢的K1部动漫,基于这些动漫做related推荐
#     @K2：在上面的related推荐结果中,留下大众打分均分最高的K2部
#     '''
#     user_topK_anime = get_user_topK_anime(train, K=K1)
#     related_recall = get_anime_related_recall()
#     u = pd.merge(user_topK_anime, related_recall, on='anime_id', how='left') \
#         .query('recall!=0') \
#         .groupby('user_id') \
#         .apply(lambda x: x.sort_values('score', ascending=False)[:10]) \
#         .drop('user_id', axis=1) \
#         .reset_index() \
#         .drop('level_1', axis=1)
#     return u
# =============================================================================
