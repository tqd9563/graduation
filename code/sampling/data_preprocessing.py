# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:49:31 2019

@author: tqd95
"""
from functools import reduce
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

user_clean = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\users_cleaned.csv')
anime_clean = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\anime_cleaned.csv')


# 逐块读取数据，细微整理
def read_useranime():
    # 一次读入100w行数据，大约需要3min
    chunkSize = 1000000
    reader = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\animelists_cleaned.csv',
                         chunksize=chunkSize,
                         usecols=['username', 'anime_id', 'my_watched_episodes', 'my_score',
                                  'my_status', 'my_last_updated'])
    df = pd.DataFrame()
    for data in reader:
        data['last_update_year'] = data['my_last_updated'].apply(
            lambda x: int(re.match('^(\d+)-(\d+)-(\d+).*$', x).group(1)))
        data['last_update_month'] = data['my_last_updated'].apply(
            lambda x: int(re.match('^(\d+)-(\d+)-(\d+).*$', x).group(2)))
        data['last_update_day'] = data['my_last_updated'].apply(
            lambda x: int(re.match('^(\d+)-(\d+)-(\d+).*$', x).group(3)))
        data.drop('my_last_updated', axis=1, inplace=True)

        int_columns = data.select_dtypes(include=['int64']).columns
        data[int_columns] = data.select_dtypes(
            include=['int64']).apply(pd.to_numeric, downcast='unsigned')
        df = pd.concat([df, data], axis=0, sort=False)
    return df


useranimelist_clean = read_useranime()


# 采样:
user_clean = user_clean.sample(frac=0.05, random_state=1001)  # 随机采样5%比例的用户，种子为1001

def sampling(df):
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


useranimelist_clean = sampling(useranimelist_clean)
useranimelist_clean.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\sampling_df.pkl')

# 初步划分训练测试集：

def train_test_split(df):
    '''
    @df: 采样后的useranimelist数据集
    返回结果:
    @train_non_zero: 最终用于模型训练的训练集,包含status=1,2,3,6的非零打分样本和statue=4的所有样本
    @train_zero：包含status=1,2,3的零分样本
    @train_plan_to_watch：包含status=6的零分样本
    @test: 不加任何处理的最原始的测试集
    '''
    df = pd.merge(df, anime_clean, on='anime_id')
    df = df.loc[:, ['user_id', 'anime_id', 'my_watched_episodes', 'episodes', 'my_score',
                    'my_status', 'last_update_date', 'title', 'aired', 'score', 'rank']]

    # 1. 以时间=2017/6/30为界，并删除p_date = 19700101的，以及my_status=5的记录，同时删除last_update_time早于动画发布日期的记录
    df = df.query('last_update_date != "1970-01-01" and my_status!=5')
    df['start'] = df['aired'].apply(lambda x: eval(x)['from'])
    df['end'] = df['aired'].apply(lambda x: eval(x)['to'])
    cond = df['last_update_date'] < df['start']
    df = df.loc[-cond, :]
    del df['aired']
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
    train.loc[((train.my_status == 6) & (
        train.my_watched_episodes > 0)), 'my_status'] = 1
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

    return train_non_zero, train_zero, train_plan_to_watch, test


train_non_zero, train_zero, train_plan_to_watch, test = train_test_split(useranimelist_clean)

train_non_zero.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_non_zero.pkl')  # 包含1236的非零分以及4
train_zero.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_zero.pkl')  # 包含123的零分
train_plan_to_watch.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_plan_to_watch.pkl')  # 包含6的0分
test.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\test.pkl')

# 用户动漫倒排表


def get_user_item(train):
    user_item = train.loc[train.label == 1, :].groupby('user_id').agg(lambda x: set(x.anime_id))
    user_item = user_item.iloc[:, 0]
    user_item = pd.DataFrame(user_item)
    user_item = user_item.reset_index()
    user_item['num'] = user_item['anime_id'].apply(lambda x: len(x))
    return user_item


user_item = get_user_item(train_non_zero)


# =============================================================================
# # =============================================================================
# # 去除异常痴迷用户(有待商榷。。因为不去掉的话实在算的太慢了。。)
# # 在计算用户物品倒排表的过程中，发现：每个用户观看的动画数，其75%分位数是114部，然而最多的能看到1645部
# # 对计算矩阵造成非常大的开销。所以决定把这部分"异常痴迷"的用户从训练集和测试集里一同去除。(异常定义为QU+1.5IQR=250)
# # 还有一个原因就是item-CF的潜在假设是：每个用户的兴趣集中在几个有限领域，因此那些看了非常多动画的人的兴趣应该非常广泛
# # 所以要过滤掉。
# # 因为发现可以计算了，，所以现在就不对这部分用户进行过滤了！
# # =============================================================================
# outlier_user_list = user_item.query('num >= 250')['user_id']
# user_item = user_item.query('num < 250')
# cond1 = train.user_id.apply(lambda u:u in outlier_user_list.values)
# cond2 = test.user_id.apply(lambda u:u in outlier_user_list.values)
# cond3 = train_plan_to_watch.user_id.apply(lambda u:u in outlier_user_list.values)
# train = train.loc[-cond1,:] # 185w->146w
# test = test.loc[-cond2,:]
# train_plan_to_watch = train_plan_to_watch.loc[-cond3,:]
# del cond1,cond2,cond3
# =============================================================================

user_item.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\user_item.pkl')


# 训练集中用户喜欢(label=1)的动画的分组频数统计：(总共4967本动画，想要进一步精简)
def get_watch_anime_freq(train):
    watched_anime_freq = train.loc[train.label == 1, :].groupby('anime_id').size()
    watched_anime_freq = watched_anime_freq.reset_index()
    watched_anime_freq = watched_anime_freq.rename(columns={0: 'times'})
    watched_anime_freq = pd.merge(watched_anime_freq, anime_clean, on='anime_id')
    watched_anime_freq = watched_anime_freq.loc[:, ['anime_id', 'times', 'title', 'aired', 'score', 'rank', 'popularity', 'favorites']]
    watched_anime_freq['start'] = watched_anime_freq['aired'].apply(lambda x: eval(x)['from'])
    watched_anime_freq['end'] = watched_anime_freq['aired'].apply(lambda x: eval(x)['to'])
    del watched_anime_freq['aired']
    return watched_anime_freq


watched_anime_freq = get_watch_anime_freq(train_non_zero)
watched_anime_set = reduce(lambda x, y: x.union(y), user_item.anime_id)
print('用户动漫正排表里有%d本动画' % len(watched_anime_set))  # 4996本动画

fig, ax = plt.subplots(1, 1)
plt.rcParams['font.sans-serif'] = ['SimHei']
sns.lineplot(x='times', y='score', data=watched_anime_freq.query('times<=50'))
ax.set_title('在训练集中y=1并且观看次数小于50次的动画，其平均得分趋势')
plt.savefig(r'C:\Users\tqd95\Desktop\graduation_thesis\less_than_50.png', dpi=300)


times_quantile = pd.DataFrame({'q': list(map(lambda x: x / 10, range(1, 10)))})
times = []
for i in range(1, 10):
    times.append(watched_anime_freq['times'].quantile(i / 10))
times_quantile['times'] = times

fig, ax = plt.subplots(1, 1)
plt.rcParams['font.sans-serif'] = ['SimHei']
sns.barplot(x='q', y='times', data=times_quantile, palette='Blues_d')
ax.set_title('训练集中y=1的动画被观看次数的分位数分布情况')
plt.savefig(r'C:\Users\tqd95\Desktop\graduation_thesis\anime_times_quantile_distribution.png', dpi=300)


# =============================================================================
# 在这4996本动画里，有近50%(2516本)动画，只被少于25个人喜欢了，并且绝大部分(2479本)动画的平均得分都低于8分
# (这个平均得分是作者给的数据，而不是我们训练集里的，不过这个得分也能反映出这个作品的受欢迎程度)
# 所以考虑在计算动画相似矩阵的时候，把这些动画也排除。不过这里我的处理方式是
# 1. 从train里过滤掉这些动画
# 2. 计算用户-动画的正排表
# 3. 算完之后，这些动画加回去用来训练模型，这些是负样本，对模型训练是有用的，只是因为算矩阵的时候所以才先删了
# 下面是更新新的user_item正排表
# =============================================================================

remained_anime_list = watched_anime_freq.query('times>25 or score>8')[['anime_id']]
train_tmp = pd.merge(train_non_zero.loc[train_non_zero.label == 1, :], remained_anime_list, on='anime_id')
new_user_item = get_user_item(train_tmp)
del train_tmp

# 过滤后剩下的2508部动画列表(也是我们用于item-CF的待选动画)：
new_watched_anime_set = reduce(lambda x, y: x.union(y), new_user_item.anime_id)
print('经过过滤后，剩下%d部动画' % len(new_watched_anime_set))  # 2508部动漫
filterd_anime_list = pd.DataFrame(new_watched_anime_set)
filterd_anime_list = filterd_anime_list.rename(columns={0: 'anime_id'})


# 保存数据集到本地
user_clean.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\sampling_user.pkl')  # 采样用户集, 比例5%
filterd_anime_list.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\filterd_anime_list.pkl')    # 生成相似矩阵用的动漫列表
new_user_item.to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\user_item.pkl')  # 生成相似矩阵用的用户-动漫倒排表


# =============================================================================
# 考虑status=1,2,3并且打分为零的样本：
# 这部分样本大概占20%, 一开始的想法是都认为负样本; 还有的思路是删了(考虑整合到特征中去);
# 或者是用最近的K个样本的平均打分来预测这个缺失的打分
# 下面的探索发现：这几个status各自缺失打分最多的30部动漫，几乎都是热门动漫！所以肯定是不能全都作为负样本的。。
# =============================================================================
def get_top_30_missing_info(status):
    df = train_zero.loc[train_zero.my_status == status, :]
    df1 = df.groupby('anime_id').size().sort_values(ascending=False)[:30]
    df1 = df1.rename('missing_times').to_frame().reset_index()
    df1 = pd.merge(df1, anime_clean.loc[:, ['anime_id', 'title', 'title_japanese']], on='anime_id')
    return df1


top_30_missing_1 = get_top_30_missing_info(1)
top_30_missing_2 = get_top_30_missing_info(2)
top_30_missing_3 = get_top_30_missing_info(3)
writer = pd.ExcelWriter(r'C:\Users\tqd95\Desktop\graduation_thesis\top_30_missing.xlsx')
top_30_missing_1.to_excel(writer, 'status=1')
top_30_missing_2.to_excel(writer, 'status=2')
top_30_missing_3.to_excel(writer, 'status=3')
writer.save()


# 分位数分析
fcut = pd.qcut(watched_anime_freq.times, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
a = watched_anime_freq.groupby(fcut)['score'].mean()
a = pd.DataFrame(a)
a = a.reset_index()
a.to_excel(r'C:\Users\tqd95\Desktop\graduation_thesis\quantile_analysis.xlsx')
