<center><font size=7>**RecSys dealing with big data**</font></center>
参考链接
[机器学习推荐或者广告场景下样本采样究竟有多重要？](https://www.zhihu.com/question/57747085)
[推荐系统正负样本的划分和采样，如何做更合理？](https://www.zhihu.com/question/334844408)
[都说数据是上限，推荐系统ctr模型中，构造正负样本有哪些实用的trick？](https://www.zhihu.com/question/324986054)

[TOC]
# 1. 机器学习推荐系统下样本采样的问题
&nbsp; &nbsp; &nbsp; &nbsp;在数据量比较大的情况下，采样是可取的。通常来说，都是正样本少，而负样本多，正负样本不平衡。
一般采用的方法是负样本下采样，可以有下面几种策略：

- 随机下采样(然而实际效果并不好，因为负样本中可能很多样本是用户都没有听说过的不能代表他不喜欢)
- **在用户未点击的部分，选择流行度高的作为负样本（更有代表性)(我觉得这个可以)**
- 在用户未点击的部分，删除用户近期已发生观看行为的电影
- 在用户未点击的部分，统计相应的曝光数据，取Top作为负样本（多次曝光仍无转化）?

&nbsp; &nbsp; &nbsp; &nbsp;确定了采样方法之后，就需要确定采样比例，**比例会影响最后模型的性能指标，建议按照比例逐个尝试，
用交叉验证的方法确定一个采样比例。**在基于上述策略生成全部的样本后，再根据时间轴，人为的划分成训练集和测试集。

&nbsp; &nbsp; &nbsp; &nbsp;其他的一些采样技巧如下：

- 不要被少数活跃用户影响了分布。从大量用户中均匀取样本。
- 尽量不要用生成样本的算法比如smote以及其各种变体，特别是对于维度很高的数据，效果非常差，高维空间下很多生成样本的算法都失效
了，因为高维空间的欧几里得距离趋于一致。
- 对每个用户保证正负样本平衡
