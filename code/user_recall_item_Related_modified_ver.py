# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:52:39 2019
修改后的item-related召回
@author: tqd95
"""
def user_related_recall(user_id, anime_stats, user_item_ranking, related_recall, user_watched_item, K=20):
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
            if cnt >= K:
                break
        if len(res) >= K:  # 只要找到了K部related 动漫就停止
            break
    return res

def get_test_related_recall(test, anime_stats, user_item_ranking, related_recall, user_watched_item, K=10):
    '''
    @test: 不加任何处理的测试集
    @user_item_ranking：用户喜欢的动漫按打分高低排序
    @related_recall: 给每一部动漫按照related字段联系到别的动漫
    @K：根据related关系,给每个用户最多召回的动漫数,默认10部
    '''
    testing = pd.DataFrame(test.user_id.unique())
    testing.columns = ['user_id']
    testing['anime_recall'] = testing['user_id'].apply(user_related_recall, anime_stats=anime_stats, user_item_ranking=user_item_ranking, 
                                                       related_recall=related_recall, user_watched_item=user_watched_item, K=K)
    #testing.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test_related_recall_10.pkl')
    return testing

test_related_recall = get_test_related_recall(test, anime_stats, user_item_ranking, related_recall, 
                                              user_watched_item, K=20)




filtered_rec_result = test_related_recall # 20

print(get_precision_recall(filtered_rec_result, test_positive))
print(get_hit_rate(filtered_rec_result, test_positive))
print(len(set(filtered_rec_result.anime_recall))/len(anime_stats))
print(get_entropy(filtered_rec_result))
print(get_type_entropy(filtered_rec_result, anime_clean))