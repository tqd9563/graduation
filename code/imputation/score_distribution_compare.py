# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:35:47 2019
填充得分分布与真实数据得分分布比较
@author: tqd95
"""

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

train_non_zero = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\sampling\train_non_zero.pkl')
item_score = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\imputation\neighborhood\item_cosine\pred_score.pkl')
user_score = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\imputation\neighborhood\user_cosine\pred_score.pkl')
knn_score = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\imputation\knn\pred_score.pkl')

# 把item和user预测的得分中,大于10分的记作10分;小于0分的记作0分
item_score.loc[item_score.my_score > 10, 'my_score'] = 10
item_score.loc[item_score.my_score < 0, 'my_score'] = 0
user_score.loc[user_score.my_score > 10, 'my_score'] = 10
user_score.loc[user_score.my_score < 0, 'my_score'] = 0


def group_score(pred_score):
    pred_score['score interval'] = pd.cut(pred_score.my_score, bins = list(range(0,11)),
                                      labels = ['('+str(i)+','+str(i+1)+']' for i in range(0,10)])

    return pred_score

item_score = group_score(item_score)
user_score = group_score(user_score)
knn_score = group_score(knn_score)
train_non_zero['score interval'] = pd.cut(train_non_zero.my_score, bins = list(range(11)),
              labels = ['('+str(i)+','+str(i+1)+']' for i in range(10)])


# plot
def score_distribution_plot(train, item_score, user_score, knn_score): 
    i = item_score.groupby('score interval').size() / item_score.groupby('score interval').size().sum()
    i = i.to_frame().reset_index().rename(columns = {0:'pct'})
    u = user_score.groupby('score interval').size() / user_score.groupby('score interval').size().sum()
    u = u.to_frame().reset_index().rename(columns = {0:'pct'})
    k = knn_score.groupby('score interval').size() / knn_score.groupby('score interval').size().sum()
    k = k.to_frame().reset_index().rename(columns = {0:'pct'})
    t = train_non_zero.groupby('score interval').size() / train_non_zero.groupby('score interval').size().sum()
    t = t.to_frame().reset_index().rename(columns = {0:'pct'})
    
    fig = plt.figure(figsize = (12,10))
    plt.rcParams['font.sans-serif'] = ['SimHei']

    ax1 = fig.add_subplot(2,2,1)
    sns.barplot(x = 'score interval', y = 'pct', data = i, palette = 'Pastel2')
    ax1.set_title('item-based方法评分分布')
    plt.ylim(0,0.4)
    ax2 = fig.add_subplot(2,2,2)
    sns.barplot(x = 'score interval', y = 'pct', data = u, palette = 'Pastel2')
    ax2.set_title('user-based方法评分分布')
    plt.ylim(0,0.4)
    ax3 = fig.add_subplot(2,2,3)
    sns.barplot(x = 'score interval', y = 'pct', data = k, palette = 'Pastel2')
    ax3.set_title('knn方法评分分布')
    plt.ylim(0,0.4)
    ax4 = fig.add_subplot(2,2,4)
    sns.barplot(x = 'score interval', y = 'pct', data = t, palette = 'Pastel2')
    ax4.set_title('原始训练集打分分布')
    plt.ylim(0,0.4)
    #plt.suptitle('图4.1：三种填充方法各自预测评分结果的分布情况')
    path = r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\imputation\pred_score.png'
    plt.savefig(path, dpi=200)
    return 

score_distribution_plot(train_non_zero,item_score,user_score,knn_score)