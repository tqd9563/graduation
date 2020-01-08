<center><font size=7>**Data spasity and Missing value imputation**</font></center>

# 1. 已有的一些工作
&nbsp; &nbsp; &nbsp; &nbsp;user-item矩阵的稀疏性问题常会导致基于邻域的协同过滤方法无法找到相应的neighbor，从而不能做出准确的推荐。为了克服这个问题，科学家们已经提出了许多方法，主要可以分成三类：

- 设计一些相似度的计算方法，使其能够允许矩阵的稀疏性[1]
- 设计更好的聚合方法，把所有neighbors的item得分信息整合在一起[2]
- 一些missing value imputation的方法，包括default voting[3], smoothing method[4], missing data prediction




# 2. 我的想法
- 首先，对于缺失打分的数据，不是所有的缺失数据都要进行填充。（因为缺失的比例挺多的）在这里，只挑选每个用户没有打分的动漫中，较为冷门的那些动漫进行imputation。
- 然后需要考虑到status，my_watched_episodes。具体地，
  1. my_watched_episodes越接近于总集数，打分的可靠性越高？
  2. status = 2的，打分可靠性高；status = 1和3的，可靠性一般，status = 4的可靠性也高（因为弃了的评分）


- 还可以尝试用该用户看过的同genre/同type之类的动漫的平均打分作为


对train_zero里的每个user看过的anime，都添加到user_watched_anime中去。

先全部填充，**然后对填充后的结果进行筛选，挑选其中有价值的样本合并入train_non_zero中去。**

1. my_watched_episodes/episodes >= 80%的样本需要预测，更低比例的话就不填充了。
2. 预测出的得分，给予一个权重，权重值就是my_watched_episodes/episodes


关于related-recall，原本只是根据用户在训练集中喜欢的动漫从高到低打分排序。可以考虑在打分相同的情况下，根据confidence降序来看？
妈的。。结果变的好差。。

那要不试试把confidence_weight作为新的特征加入模型训练中去。

可以画一张图，比较真实存在的打分分布情况，与imputation的打分分布情况，看看二者是否近似！！(doged bar plot)

有些用户质量很低。。比如User_id = 553，打分全是0，status几乎都是4

对于用户邻域的方法，原始版本是直接算u的打分均值。可以考虑一下限制在同类动漫下的打分均值吗？

-----------------------------------重要！！！------------------------------------------
对于物品邻域的方法，因为是要找目标物品j的半径为K的邻域，因此物品j必须和其他动漫有相似度的计算。也因此，物品j必须
在train_non_zero中，有人打分过。
所以，最后预测的时候，target_zero需要再经过一次筛选，选出anime_id在train_non_zero里出现过的，而且得是非零的打分。
经过排查发现，target_zero里有44部动漫，是不在train_non_zero的非零打分集中的。其中：
- 有2部在train_non_zero里出现过，但是都是零分
- 另外42部是没有在train_non_zero里有任何的打分记录
这44部动漫大约占据了target_zero的多少行呢？区区58行，直接删了

-----------------------------------------
基于item的邻域方法预测得分，选用了两种item间的相似度：cosine和pearson
对比这两种wij的方法得到的结果，大概可以认为：cosine的表现略优于pearson方法。
对比item(cosine)方法和原始的sampling结果，全面提升。
对比item(cosine)方法和KNN的比较，互有胜负。










参考文献
[1] C. Desrosiers and G. Karypis. A Novel Approach to Compute Similarities and Its Application to Item
Recommendation. In Proceeding of PRICAI 2010, pages 39–51, 2010
[2] M. Larson and A. Hanjalic. Exploiting User Similarity based on Rated-Item Pools for Improved User-based
Collaborative Filtering. In Proceedings of the 3rd ACM Conference on Recommender Systems, pages 125–132, 2009
[3]  J. Breese, D. Heckerman, C. Kadie, and Others. Empirical analysis of predictive algorithms for
collaborative filtering. In Proceedings of the 14th conference on Uncertainty in Artificial Intelligence,
pages 43–52, 1998.
[4]  G.-R. Xue, C. Lin, Q. Yang, W. Xi, H.-J. Zeng, Y. Yu, and Z. Chen. Scalable collaborative filtering using
cluster-based smoothing. In SIGIR 2005, pages 114–121, New York, New York, USA, 2005. ACM Press.