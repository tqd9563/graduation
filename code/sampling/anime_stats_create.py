# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:34:27 2019
生成动漫画像特征
@author: tqd95
"""

import pandas as pd

anime_clean = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\anime_cleaned.csv')
filterd_anime_list = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\filterd_anime_list.pkl')    # 生成相似矩阵用的动漫列表
# a1就是我们的动漫画像骨架
a1 = anime_clean.loc[:,['anime_id','title_japanese','type','source','episodes',
                        'aired','rating','score','scored_by','members','favorites',
                        'related','studio','genre','aired_from_year','duration_min']]


# episodes+duration_min处理
## 新添字段anime_type,表示泡面番、短篇、季番、半年番、长篇动画(对TV type的补充), 其余type保持不变
episodes = a1.groupby('episodes').size().sort_values(ascending=False) # 出现次数最多的是1, 以Movie和OVA, Special居多
a1.loc[((a1.type == "TV") & (a1.episodes >= 55)),'anime_type'] = 'Long TV'
a1.loc[((a1.type == "TV") & (a1.episodes >= 30) & (a1.episodes < 55)),'anime_type'] = 'Year TV'
a1.loc[((a1.type == "TV") & (a1.episodes >= 14) & (a1.episodes < 30)),'anime_type'] = 'Half year TV'
a1.loc[((a1.type == "TV") & (a1.episodes >= 10) & (a1.episodes < 14)),'anime_type'] = 'Quarter TV'
a1.loc[((a1.type == "TV") & (a1.duration_min >= 3) & (a1.duration_min<=6)),'anime_type'] = 'Instant TV' # 泡面番
a1.loc[((a1.type == "TV") & (a1.episodes !=  0) & (a1.episodes < 10)),'anime_type'] = 'Short TV'
## 剩余的五种type保持不变。
a1.loc[a1.type == "OVA",'anime_type'] = 'OVA'
a1.loc[a1.type == "Special",'anime_type'] = 'Special'
a1.loc[a1.type == "ONA",'anime_type'] = 'ONA'
a1.loc[a1.type == "Movie",'anime_type'] = 'Movie'
a1.loc[a1.type == "Music",'anime_type'] = 'Music'
del a1['type']
## 还剩下61部TV动画，他们的episodes = 0，手动排查一下吧。。


# source处理
## 把Music, Book, Picture book, Radio和Digital manga合并到others类; 把Card game合并到Game里
## Light novel和Novel不知道可不可以合在一起?(暂时没这么做)
source = a1.groupby('source').size().sort_values(ascending=False) 

a1.loc[a1.source == 'Picture book','source'] = 'Other'
a1.loc[a1.source == 'Book','source'] = 'Other'
a1.loc[a1.source == 'Music','source'] = 'Other'
a1.loc[a1.source == 'Radio','source'] = 'Other'
a1.loc[a1.source == 'Digital manga','source'] = 'Other'
a1.loc[a1.source == 'Card game','source'] = 'Game'

# aired处理
## 拆分出动漫的始末日期：
a1['start'] = a1['aired'].apply(lambda x:eval(x)['from'])
a1['end'] = a1['aired'].apply(lambda x:eval(x)['to'])
del a1['aired']

# scored_by处理
## 把scored_by除以members, 得到将这部动画加入到list的用户中有多少比例的用户最后打了分。
a1['scored_ratio'] = a1['scored_by']/a1['members']
del a1['scored_by']


# members处理
## members的跨度非常大,有85%的样本是小于10w的,有40%的样本是小于1w的
## 初步的想法是按下面的区间划分：0~1k, 1k~5k, 5k~2w, 2w~10w, 10w~30w, 30w+
## 因为是含有等级的，后面要用label encoding!!
members = a1.members
members_quantile = pd.DataFrame({'q':list(map(lambda x:x/20,range(1,20)))})
res = []
for i in range(1,20):
    res.append(a1.members.quantile(i/20))
members_quantile['members']=res 


bins=[0, 1000, 5000, 20000, 100000, 300000, 1500000]
labels = ['Unpopular','Not so popular','Common','A bit popular','Quite popular', 'Extremely popular']
a1['popularity'] = pd.cut(a1['members'],bins=bins,labels=labels)

# studio处理
## 出现过的制作公司有好多,但是可以只挑选其中常见的一些，剩下的就用others代替吧。不然维度太高了
studio = a1.groupby('studio').size().sort_values(ascending=False)
high_freq_studio = studio[:15]
cond = a1.studio.apply(lambda x:x not in high_freq_studio)
a1.loc[cond,'studio'] = 'Other'


# genre处理
def get_anime_genre(total_anime_list):
    '''
    @total_anime_list: 全动漫信息
    返回结果: 每个anime_id对应的genre,一个anime_id有多行
    '''
    g = total_anime_list.loc[:,['anime_id','genre']]
    g['genre'].fillna('',inplace=True)
    genre = g['genre'].apply(lambda x:x.split(',')).apply(pd.Series).stack().rename('genre')
    genre = genre.to_frame().reset_index(1,drop=True)
    genre['genre'] = genre['genre'].apply(lambda s:s.strip())
    genre = genre.join(g.anime_id).reset_index(drop=True)
    anime_genre = genre.loc[:,['anime_id','genre']]
    return anime_genre

anime_genre = get_anime_genre(anime_clean)

def topK_anime_genre_ranking(filterd_anime_list, total_anime_list, K=20):
    '''
    @filterd_anime_list: 过滤后的动漫,即训练集中用户喜欢的所有动漫
    @total_anime_list: 全量动漫信息
    输出结果：返回的是过滤动漫集中的top20 genre排名，以及这20个genre在全部6700部动漫中的排名(排名就是频率)
    '''
    a1 = total_anime_list.loc[:,['anime_id','genre']]
    a1['genre'].fillna('',inplace=True)
    all_anime_genre_rank = a1['genre'].apply(lambda x:x.split(',')) \
                                .apply(pd.Series).stack() \
                                .reset_index(1,drop=True) \
                                .apply(lambda s:s.strip()) \
                                .value_counts().rename('counts_in_all_anime') \
                                .to_frame() # 这个算的是所有动漫的genre排名,index是对应的genre
    all_anime_genre_rank['rank_in_all_anime'] = list(range(1, len(all_anime_genre_rank)+1))
    
    anime_genre = get_anime_genre(total_anime_list)
    genre_top_K_in_train = pd.merge(filterd_anime_list,anime_genre,on='anime_id') \
                                .groupby('genre').size() \
                                .sort_values(ascending=False)[:K] \
                                .rename('counts_in_train') \
                                .to_frame()
    genre_top_K_in_train['rank_in_train'] = list(range(1, K+1))
    genre_top_K_in_train = genre_top_K_in_train.join(all_anime_genre_rank['rank_in_all_anime'])
    return genre_top_K_in_train


genre_top_20_in_train = topK_anime_genre_ranking(filterd_anime_list,anime_clean,K=20)


## 决定保留这20个出现频率最高的genre, 其余的genre就用Other来代替
def topK_anime_genre_onehot_encoding(filterd_anime_list, total_anime_list, K=20):
    '''
    @filterd_anime_list: 过滤后的动漫,即训练集中用户喜欢的所有动漫
    @total_anime_list: 全量动漫信息
    输出结果：每个anime_id对应的K+1个genre的onehot信息。一部动漫只占一行
    '''
    anime_genre = get_anime_genre(total_anime_list)   
    genre_top_K_in_train = topK_anime_genre_ranking(filterd_anime_list, total_anime_list, K=K)
    
    # 把全量动漫中那些不是训练集热门topK的genre替换为Other
    cond = anime_genre['genre'].apply(lambda x:x not in genre_top_K_in_train.index)
    anime_genre.loc[cond,'genre'] = 'Other'
    
    # 判断：如果一个动画的genre全是Other,则去重;如果一个动画的genre除了Other以外还有别的,那就把Other删了
    anime_genre = anime_genre.groupby('anime_id').apply(lambda df:set(df.genre)) \
                    .rename('genre').to_frame() \
                    .reset_index()
    def drop_other(s):
        if s == {'Other'}: pass
        elif 'Other' in s:
            s = s - {'Other'}
        else: pass
        return s
    anime_genre['genre'] = anime_genre['genre'].apply(drop_other)
    anime_genre['genre'] = anime_genre['genre'].apply(lambda x:list(x)) # 到这里,只需要explode一下就行了
    def explode(df,group_by_field,explode_field):
        df1 = df[explode_field].apply(pd.Series).stack().rename(explode_field)
        df2 = df1.to_frame().reset_index(1,drop=True)
        res = df2.join(df[group_by_field]).reset_index(drop=True)
        return res.loc[:,[group_by_field,explode_field]]
    
    anime_genre = explode(anime_genre,'anime_id','genre')
    anime_genre = pd.get_dummies(anime_genre).groupby('anime_id').max().reset_index()
    return anime_genre


genre_ohe = topK_anime_genre_onehot_encoding(filterd_anime_list,anime_clean,K=20)
a1 = pd.merge(a1,genre_ohe,on='anime_id')
a1.drop('genre',axis=1,inplace=True)
a1.columns = pd.Index(['a_'+c for c in a1.columns])  # 给动漫画像特征都加上前缀'a_'用以区分
a1.loc[:,['a_anime_id','a_source','a_anime_type','a_rating','a_score','a_scored_ratio',
          'a_start','a_end','a_populatiry','a_studio','a_genre_Action',
          'a_genre_Adventure', 'a_genre_Comedy', 'a_genre_Drama', 'a_genre_Ecchi',
          'a_genre_Fantasy', 'a_genre_Harem', 'a_genre_Magic', 'a_genre_Mecha',
          'a_genre_Military', 'a_genre_Mystery', 'a_genre_Other',
          'a_genre_Romance', 'a_genre_School', 'a_genre_Sci-Fi', 'a_genre_Seinen',
          'a_genre_Shoujo', 'a_genre_Shounen', 'a_genre_Slice of Life',
          'a_genre_Super Power', 'a_genre_Supernatural']] \
          .to_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\anime_stats.pkl')
