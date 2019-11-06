# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:55:34 2019
评价模型在测试集上的效果
@author: tqd95
"""
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读入测试集和模型
test_fn = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test_fn.pkl')
test_recall_mapping = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test_recall_channel_mapping.pkl')
test = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test.pkl') 
anime_stats = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\anime_stats.pkl')
anime_clean = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\anime_cleaned.csv')

with open('xgboost.pkl', 'rb') as fr:
    model = pickle.load(fr)

# 特征重要性
df = pd.DataFrame({'feature':test_fn.columns,
                   'importances':model.feature_importances_})
df = df.sort_values(by='importances',axis=0,ascending=False).iloc[:20,:]


fig,ax=plt.subplots(1,1)
plt.rcParams['figure.figsize'] = (12.0, 5.0)
plt.rcParams['font.sans-serif'] = ['SimHei']
sns.barplot(y='feature',x='importances',data=df,orient='h')
ax.set_title('XGBoost特征重要性排名前20')
fig.subplots_adjust(left=0.45)
plt.savefig(r'C:\Users\tqd95\Desktop\graduation_thesis\result\feature_importances',dpi=300)







# 把训练好的模型应用在最终测试集上,返回一个经过过滤后的结果
# 过滤规则是：对每一个用户,召回得分最高的10部新番和20部旧番
def get_recommend_result(model, test_fn):
    '''
    @model: 训练好的XGBoost排序模型
    @test_fn: 最终测试集, 可以用来预测打分的(经过了onehot处理的)
    '''
    # test_recall_mapping: 各类召回算法的综合结果,包含四列：user_id,anime_recall,recall_channel和is_new_anime
    test_recall_mapping = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test_recall_channel_mapping.pkl')
    y_pred = model.predict_proba(test_fn)[:,1]
    # 模型在测试集上的预测得分情况(预测样本的label=1的概率)
    test_recall_result = pd.concat([test_recall_mapping, pd.DataFrame(y_pred)], axis = 1) \
                            .rename(columns = {0:'pred_score'})
    
    # 定义一个函数,针对同一个user召回的旧番还是新番,分别召回K1和K2部
    def get_rec_result(df, K1=20, K2=10):
        df['ranking'] = df.groupby('is_new_anime')['pred_score'].rank(ascending = False, method = 'first')
        res = df.loc[((df.is_new_anime==0) & (df.ranking<=K1)) | ((df.is_new_anime==1) & (df.ranking<=K2)),:] \
            .sort_values(by = ['is_new_anime','ranking'], ascending=[True,True])
        return res

    res = test_recall_result.groupby('user_id').apply(get_rec_result, K1=20, K2=10)
    res = res.drop('user_id',axis=1) \
            .reset_index() \
            .drop('level_1',axis=1)
    res['recall_channel'] = res['recall_channel'].apply(lambda x:x[0])
    res['anime_recall'] = res['anime_recall'].astype(int)
    res['is_new_anime'] = res['is_new_anime'].astype(int)
    res['ranking'] = res['ranking'].astype(int)
    return res

filtered_rec_result = get_recommend_result(model, test_fn)



# 选出test里最终每个用户看的动漫,即my_watched_episodes>0的动漫, 并标注是新番还是旧番
# 这个结果存在test_positive里，也是后续和推荐结果进行比较的基准
cond = test['start'] > '2017-06-30'
test.loc[cond,'is_new_anime'] = 1
test.loc[-cond,'is_new_anime'] = 0
test_positive = test.loc[test.my_watched_episodes > 0,['user_id','anime_id','my_score','my_status','is_new_anime']]


# 计算精准率和召回率
def get_precision_recall(filtered_rec_result, test_positive):
    '''
    @filtered_rec_result: 模型经过排序、过滤后的结果
    @test_positive: 测试集里的正样本,即每个用户最终观看了的动漫, 有标记是旧番还是新番
    '''
    rt = 0
    t = 0
    r = 0
    for uid in set(test_positive.user_id):
        tu = set(test_positive.loc[test_positive.user_id == uid, 'anime_id'])
        ru = set(filtered_rec_result.loc[filtered_rec_result.user_id == uid, 'anime_recall'])
        rt += len(ru & tu)
        r += len(ru)
        t += len(tu)
    recall = rt/t       # 召回率 0.06935287372796331
    precision = rt/r    # 精准率 0.10402418542156533
    return precision, recall

precision, recall = get_precision_recall(filtered_rec_result, test_positive)


# 计算命中率。定义为：如果某个用户的推荐列表里有动漫被看了，则命中。
def get_hit_rate(filtered_rec_result, test_positive):
    '''
    @filtered_rec_result: 模型经过排序、过滤后的结果
    @test_positive: 测试集里的正样本,即每个用户最终观看了的动漫, 有标记是旧番还是新番
    '''
    cnt = 0
    for uid in set(test_positive.user_id):
        tu = set(test_positive.loc[test_positive.user_id == uid, 'anime_id'])
        ru = set(filtered_rec_result.loc[filtered_rec_result.user_id == uid, 'anime_recall'])
        if tu & ru:
            cnt += 1
    hit_rate = cnt/len(set(test_positive.user_id))
    return hit_rate

hit_rate = get_hit_rate(filtered_rec_result, test_positive)

# 计算覆盖率(最原始的,即最后推荐的物品数/总的物品数)
coverage = len(set(filtered_rec_result.anime_recall))/len(anime_stats)

# 计算item分布的覆盖率
def get_entropy(filtered_rec_result):
    # 计算最终推荐列表中所有推荐物品的分布的信息熵(对数是以2为底的)
    df = filtered_rec_result.groupby('anime_recall').size().rename('cnt').to_frame()
    df['prob'] = df['cnt']/sum(df['cnt'])
    entropy = -sum(df['prob'] * (df['prob'].apply(lambda x:np.log2(x))))
    return entropy

entropy = get_entropy(filtered_rec_result)

# 计算基于type的覆盖率
def get_type_entropy(filtered_rec_result, anime_clean):
    df = pd.merge(filtered_rec_result, anime_clean.loc[:,['anime_id','type']], left_on = 'anime_recall', right_on = 'anime_id', how = 'left')
    df = df.drop('anime_id',axis = 1)
    res= df.groupby('type').size().rename('cnt').to_frame()
    res['prob'] = res['cnt']/sum(res['cnt'])
    entropy = -sum(res['prob'] * (res['prob'].apply(lambda x:np.log2(x))))
    return entropy
type_entropy = get_type_entropy(filtered_rec_result, anime_clean)




## 看看最终推荐的结果里，有没有不在filterd_anime_list里，但是也不是新番的旧番？？
filterd_anime_list = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\filterd_anime_list.pkl')
s1 = set(filterd_anime_list.anime_id)
s2 = set(anime_stats.a_anime_id) - set(anime_stats.query('a_start>"2017-06-30"').a_anime_id) # 所有老番
s3 = s2 - s1 # 不是新番，也不是那过滤下来的2500部老番
len(set(filtered_rec_result.anime_recall) & s3)  # 只有80部


## 冲撞率(推荐的结果在用户的0分集合内(status=1,2,3,6),计算方式类比召回/精准)
def get_confict_rate(filtered_rec_result, train_zero):
    '''
    @filtered_rec_result: 模型经过排序、过滤后的结果
    @test_positive: 测试集里的正样本,即每个用户最终观看了的动漫, 有标记是旧番还是新番
    '''
    rt = 0
    t = 0
    #r = 0
    for uid in set(train_zero.user_id):
        tu = set(train_zero.loc[train_zero.user_id == uid, 'anime_id'])
        ru = set(filtered_rec_result.loc[filtered_rec_result.user_id == uid, 'anime_recall'])
        rt += len(ru & tu)
        #r += len(ru)
        t += len(tu)
    conflict_rate = rt/t       # 召回率 0.06935287372796331
    return conflict_rate

train_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_zero.pkl') # 包含123的零分
train_plan_to_watch = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_plan_to_watch.pkl') # 包含6的0分
conflict_rate = get_confict_rate(filtered_rec_result, train_zero)
conflict_rate_2 = get_confict_rate(filtered_rec_result, train_plan_to_watch)


import time
start = time.time()
test_recall = get_test_recall(test=test, user_item=user_item, user_watched_item=user_watched_item, W=W, K1=10, K2=10)
end = time.time()
print('time used:', (end-start)/60, 'min')




# 单路召回测试
get_precision_recall(filtered_rec_result, test_positive)
get_hit_rate(filtered_rec_result, test_positive)
len(set(filtered_rec_result.anime_recall))/len(anime_stats)
get_entropy(filtered_rec_result)
get_type_entropy(filtered_rec_result, anime_clean)
get_confict_rate(filtered_rec_result, train_zero)
get_confict_rate(filtered_rec_result, train_plan_to_watch)










