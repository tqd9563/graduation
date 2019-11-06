# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 08:53:00 2019
填充方法的得分分布与真实分布的对比
@author: tqd95
"""

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_non_zero.pkl')
cs = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\imputation\neighborhood\item_cosine\pred_score.pkl')
ps = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\imputation\neighborhood\item_pearson\pred_score.pkl')

def func(pred_score):
    mm = MinMaxScaler()
    res = 10 * mm.fit_transform(pred_score)
    pred_score['my_score_scaled'] = res.reshape(len(res))
    pred_score['group_raw'] = pd.cut(pred_score.my_score, bins = list(range(0,13)),
                                      labels = ['('+str(i)+','+str(i+1)+']' for i in range(0,12)])
    
    pred_score['group_scaled'] = pd.cut(pred_score.my_score_scaled, bins = list(range(11)),
                                          labels = ['('+str(i)+','+str(i+1)+']' for i in range(10)])
    
    return pred_score

cs = func(cs)
ps = func(ps)
    

def score_distribution_plot(train, cosine_pred, pearson_pred, method = 'raw'):
    group_name = 'group_' + method
    cosine = cosine_pred.groupby(group_name).size() / cosine_pred.groupby(group_name).size().sum()
    cosine = cosine.to_frame().reset_index().rename(columns = {0:'pct'})
    pearson = pearson_pred.groupby(group_name).size() / pearson_pred.groupby(group_name).size().sum()
    pearson = pearson.to_frame().reset_index().rename(columns = {0:'pct'})
    
    train['group'] = pd.cut(train.my_score, bins = list(range(11)),
                             labels = ['('+str(i)+','+str(i+1)+']' for i in range(10)])
    
    t = train.groupby('group').size() / train_non_zero.groupby('group').size().sum()
    t = t.to_frame().reset_index().rename(columns = {0:'pct'})
    
    fig = plt.figure(figsize = (15,5))
    ax1 = fig.add_subplot(1,3,1)
    sns.barplot(x = group_name, y = 'pct', data = cosine, palette = 'Set2')
    plt.ylim(0,0.5)
    plt.xticks(rotation=-45)
    ax1.set_xlabel('item_cosine_' + method)
    ax2 = fig.add_subplot(1,3,2)
    sns.barplot(x = group_name, y = 'pct', data = pearson, palette = 'Set2')
    plt.ylim(0,0.5)
    plt.xticks(rotation=-45)
    ax2.set_xlabel('item_pearson_' + method)
    ax3 = fig.add_subplot(1,3,3)
    sns.barplot(x = 'group', y = 'pct', data = t, palette = 'Set2')
    plt.ylim(0,0.5)
    plt.xticks(rotation=-45)
    ax3.set_xlabel('training')
    return 


score_distribution_plot(train_non_zero, cs, ps, method = 'raw')








