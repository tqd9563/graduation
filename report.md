<center><font size=7>**Report**</font></center>

[TOC]
# 1. 推荐系统概述
## 1.1 什么是推荐系统
&nbsp; &nbsp; &nbsp; &nbsp;在这个信息过载的时代，我们作为信息的消费方，在没有明确选择意向的情况下，想要从大量的信息中找到自己可能感兴趣的信息是一件不容易的事情。反过来，供应商/平台作为信息的生产方，如何让自己生产的信息受到广大消费者的关注也是很困难的。而推荐系统的任务就是把用户和信息联系在一起，帮助用户发现自己感兴趣的信息，同时让信息出现在会对它感兴趣的用户面前。**从本质上来说，推荐系统就是在做用户行为模式挖掘，通过分析用户的历史行为数据，找出用户的行为特征，然后给出相应的预测结果。**

## 1.2 推荐系统的基本流程
&nbsp; &nbsp; &nbsp; &nbsp;一个普通的推荐系统通常可以分成下面三个阶段：

- 数据的采集与处理
- 召回
- 排序

### 1.2.1 采集和处理数据
&nbsp; &nbsp; &nbsp; &nbsp;推荐系统是在做用户行为模式挖掘，而这个挖掘是通过分析用户对物品的行为数据来进行的。用户行为数据最普遍的存在形式就是日志，网站会把用户产生的各种行为记录在日志中。比如在电商网站，用户的行为可以有浏览网页、点击、购买、收藏、评论、评分等等。我们的推荐系统模型所学习分析的用户行为数据，应当尽可能地覆盖用户可能涉及到的业务流程。

&nbsp; &nbsp; &nbsp; &nbsp;除了用户行为数据外，还需要准备的数据就是物品相关的属性数据，这部分数据也应当尽可能地覆盖更多的属性维度。物品的信息越全，模型最后作出的推荐结果也会越准确。

### 1.2.2 召回
&nbsp; &nbsp; &nbsp; &nbsp;通常推荐模型的计算开销会比较大，而我们可供推荐的物品集大小往往不下于上万种。如果完全依赖模型进行推荐的话，就要对着至少万级个物品一个一个排序打分，明显成本太高了。这时候就需要设计召回策略，将用户可能感兴趣的物品从全量的物品库中事先提取出来，作为我们推荐内容的候选集，最终为每个用户推荐的结果都会是这个候选集的某一个子集，这个候选集的大小一般不会很大。

&nbsp; &nbsp; &nbsp; &nbsp;召回的算法是非常丰富的，比如有热门推荐、基于内容/模型/标签等的召回、协同过滤召回、主题模型等等。通常来说，单一的召回算法得出的结果比较难以满足业务需求，因此**往往采用多路召回的策略**，即：每一路召回尽量采取一个不同的策略，各拉取K条物料，这个K值可以根据各路召回算法的实际表现分配不同的大小。

### 1.2.3 排序
&nbsp; &nbsp; &nbsp; &nbsp;排序阶段针对上面多路召回的结果候选集，利用排序算法进行逐一打分和重排序，以更好的反映用户的偏好。通过排序优化用户对召回集的点击行为后，将用户更可能喜欢的物品（预测打分较高的物品，通常取前N个）取出来推荐给用户（topN推荐），提升用户体验。排序算法的种类也很丰富，比如有LR,GBDT等机器学习模型，有BPR等排序模型，还有Wide & Deep, DNN等深度学习模型等等。

# 2. 推荐系统的发展历史与文献综述
&nbsp; &nbsp; &nbsp; &nbsp;最早的推荐系统大概可以追溯到1994年，明尼苏打大学的GroupLens研究组设计了第一个自动化的新闻推荐系统GroupLens[1]。这个系统不但首次提出了协同过滤的思想，而且为后世的推荐问题建立了一个形式化的范式。顺便一提这个研究组后来创建了大名鼎鼎的MovieLens推荐网站。1997年，Resnick 等人[2]首次提出推荐系统（recommendersystem，RS）一词，自此，推荐系统一词被广泛引用，并且推荐系统开始成为一个重要的研究领域。1998年，著名的个性化商城Amazon推出了基于项目的协同过滤算法(item-based Collaborative Filtering)，在此之前所普遍使用的协同过滤算法都是基于用户的(user-based CF)，而亚马逊提出的这个新算法的实际效果非常好。2003年，Amazon在IEEE Internet Computing[3]上公开了这个item-CF算法，带来了广泛的关注和使用，包括YouTube、Netflix等。2005年Adomavicius等人的综述论文[4] 将推荐系统分为3个主要类别，即基于内容的推荐、基于协同过滤的推荐和混合推荐的方法，并提出了未来可能的主要研究方向。

&nbsp; &nbsp; &nbsp; &nbsp;到了2006年，一个大事件将推荐系统的研究推向了快速发展的高潮阶段：北美在线视频服务提供商Netflix宣布了一项竞赛，第一个能将现有推荐算法的准确度提升10%以上的参赛者将获得100万美元的奖金。这个比赛在学术界和工业界引起了很大的关注，吸引了来自186个国家和地区的超过4万支队伍参赛。而在之后的那几年，许多经典的推荐算法被提出，比如：Koren等人[5]对利用矩阵分解（Matrix Factorization, MF）实现协同过滤的现有技术进行了综述，包括基本MF原理，以及包含隐反馈、时序动态等特殊元素的MF等；Steffen Rendle[6]等人在2009年提出了一种“基于贝叶斯后验优化”的个性化排序算法BPR，它是采用pairwise训练的一种纯粹的排序算法；Koren大神在2008年发表的论文[7]中，对两种较成功的CF算法：隐语义模型(LFM)和基于邻域的模型(neighbourhood-based model)进行了一定的创新与融合，提出了著名的SVD++算法。steffen rendle[8]在他2010年发表的论文中提出了大名鼎鼎的因子分解机算法（Factorization Machines
,FM）它可以直接引入任意两个特征的二阶特征组合，其本质上是在对特征进行embedding化表征，和目前非常常见的各种实体embedding本质思想是一脉相承的。FM可以很好的处理稀疏数据。

&nbsp; &nbsp; &nbsp; &nbsp;近年来随着人工智能的火爆，机器学习和深度学习技术也被广泛运用到了推荐系统的排序场景中。Google在2016年最新发布的模型Wide&Deep[9]将Logistic Regression和forward DNN结合在一起，既发挥了逻辑回归的优势，又将dnn和embedding的自动特征组合学习和强泛化能力进行了补充。这个模型在Google play的推荐场景取得了应用并且获得良好的效果；Guo, Huifeng[10]等人于2017年提出了DeepFM模型，将传统的因子分解机FM和深度神经网络DNN结合在一起，用于解决CTR预估中，挖掘构造有用的高级交叉特征的问题。

&nbsp; &nbsp; &nbsp; &nbsp;除了召回算法和排序算法的不断进步外，关于推荐系统的可解释性、冷启动、结果多样性、显隐式反馈的应用等研究也在持续推进中，也涌现出了不少优秀的方法。比如，Xiang Wang[11]等人在2018年提出了一种基于树模型的可解释推荐TEM，利用树模型得到有效的交叉特征，然后把这些交叉特征放入基于embedding的attention模型中；Hong-Jian Xue[12]等人提出了一种新的基于神经网络的矩阵分解模型，同时考虑了显式反馈和隐式反馈，将其融合构建成一个矩阵，从而将用户和产品的不同向量输入到两个并行的深层网络中去。最后，设计了一种新型的损失函数以同时考虑评分和交互两种不同类型的反馈数据；Frolov E等人针对用户喜好行为不充分的数据，对标准SVD进行了改进，提出了一种可以联合分解用户行为和物品辅助信息的HybridSVD方法，保持了原有SVD分解方法的算法简便性，并且可以一定程度上解决冷启动问题；Zhi-Peng Zhang等人[14]针对item-CF中的完全物品冷启动问题，提出了一种非常简单的通过挖掘关联属性的方法，很好的应对了冷启动问题。

# 3. 实验：利用机器学习技术和一些统计方法在MyAnimeList数据集上做动漫推荐
## 3.1 数据集介绍
&nbsp; &nbsp; &nbsp; &nbsp;这是在Kaggle上公开的一个dataset，记录了作者从网站[MyAnimeList.net](MyAnimeList.net)上爬取的用户添加动漫到list、观看、并进行打分的一系列信息。Kaggle上也有其他类似的数据集，不过它们的数量级小很多。这个数据集旨在成为对互联网御宅社区进行人口统计分析和对该群体内部的趋势分析时的代表性样本。

&nbsp; &nbsp; &nbsp; &nbsp;整个数据集可以分成三个部分：用户信息、动漫信息、以及用户给动漫打分的信息，原始数据分别存储在UserList.csv, AnimeList.csv和UserAnimeList.csv中。这个数据集的公开者已经对这些数据进行了一定的预处理，分为两个步骤，产生的结果分别是Filter数据集和Clean数据集。

- Filtered数据集：主要是把全量用户信息中，gender、birth_date、location这三个特征中凡是有缺失的用户统统剔除后所得。在删除这些用户时，动漫信息表中的一些统计特征，诸如score、popularity、members等仍然保持不变。这部分清理是针对user进行的，经过处理后，用户信息表的大小从最开始的30.2w用户缩小到了11.6w用户。
- Cleaned数据集：在Filtered数据集的基础上，主要是清理一些异常值，比如除去年龄太大或是太小的用户、修正了一些观看集数的异常、删除studio和source缺失的动漫、未播出的动漫等。这部分清理主要是针对anime进行的，经过处理后，动漫信息表的大小从最开始的14474行缩小到了6668行。

&nbsp; &nbsp; &nbsp; &nbsp;下面介绍一下各个数据集的一些主要特征(后面会做加工处理):

- 用户信息表(user_cleaned): 主要有用户正在观看的动漫数、已看完的动漫数、放弃的动漫数、总观看时长(天计)、性别、地区、出生日期、注册时间、最后一次在线时间、平均打分等
- 动漫信息表(anime_cleaned)：主要有动漫类型、原作类型、集数、上映状态、上映始末日期、适合人群、平均得分、添加到list的人数、添加到最爱的人数、相关动画、制作公司、流派等
- 用户打分行为表(useranimelist)：主要有用户名、动画id、观看的集数、最后一次状态更新日期、我的打分、我的状态

## 3.2 实验的设计
### 3.2.1 清洗数据和采样
&nbsp; &nbsp; &nbsp; &nbsp;对于下载下来的Cleaned数据集，我还做了以下一些处理。

- 用户行为数据集中有一个字段是my_status，表示的是用户当前的状态，在作者的说明文件里指出，只有status=1, 2, 3, 4, 6的记录是有意义的，别的取值意义不明，所以这里我把其他的记录删了。另外，status=2对应的是已看完，而status=6对应的是打算观看，所以对数据集中，所有已观看集数=动漫集数的样本的status都令其等于2，而一些本来status=6，但是已经观看了一些集数的样本，status令其等于1(表示正在观看中)。
- last_update_date字段有一部分样本取值为"1970-01-01"，这部分记录删除。同时还删去了一部分last_update_date取值小于动漫开始日期的样本(明明还没上映，却有了状态，属于异常行为)

&nbsp; &nbsp; &nbsp; &nbsp;另外，由于数据集本身过大，从效率上考虑，我做了一定的采样，随机选择了5%的用户数据（大约5400名用户的90万行数据）

### 3.2.2 训练集与测试集的划分和结构
&nbsp; &nbsp; &nbsp; &nbsp;首先要说的是，我想做的是一个topN推荐而不是评分预测。简单地说，对某一个用户，我推荐了N部动漫给她，我希望的结果是：这N部动漫尽可能多的被这个用户观看了，而不是想要用户对这N部动漫的评分和我预测的评分差距不大。Amazon著名前科学家Greg Linden曾发表过一篇文章，文章里提出电影推荐的目的是找出用户最可能产生兴趣的电影，而不是来预测用户会对这部电影打多少分。动漫和电影是相似的物品，因此我的实验里，topN推荐更符合实际需求，我想要预测出每个用户最可能想看什么动漫。

&nbsp; &nbsp; &nbsp; &nbsp;数据集里有一个last_update_date字段，表示用户最后一次状态更新的时间，只要对用户anime list中的某一部动漫有发生行为，比如打分、更新status、更新watched episodes等，都会更新这个时间。整个数据集的时间范围是2006-10-01到2018-05-20。**我以2017-06-30为时间界限**，时间早于这个行为数据的作为训练集，时间晚于这个的行为数据作为测试集。

&nbsp; &nbsp; &nbsp; &nbsp;关于训练集的标签y，因为我做的是topN推荐，所以更倾向于是分类问题而不是回归问题。具体地，**我综合了训练样本里的my_status和my_score两个特征来决定样本的标签y：**

- 如果用户的打分大于等于8分，则认为用户对这部动漫感兴趣，y=1；有打分信息但是打分低于8分的，则令y=0
- 对于status=4("dropped")来说，这个状态表示的是放弃观看。所以那些打分大于等于8分的，我可以认为用户对这部动漫一开始是感兴趣的，或者说看到一半还是感觉不错的，只是中途剧情突然不合胃口，或是什么别的原因导致弃番，但是还是可以认为用户对这类动漫会有看的兴趣。而那些打分低于8分的则一律令y=0
- 对于status=6("planned to watch")，这个状态表示打算观看但是还没看，说明用户会有一定的兴趣，**这部分样本我最初的选择把这部分待观看的动漫中，没有打分信息的那部分加工成用户相关的特征，并不保留在训练集中。**如果有打分的则遵从前面的标准。
- 剩下的没有标记y的样本，就是status=1,2,3的缺失打分信息的样本了。**这部分样本暂时先从训练集里抽出来，后续再考虑处理。**

上面对y的处理，我的想法主要是这样的：训练集中的y=1表示我告诉模型用户会想看这部动漫（或者这类动漫）, y=0表示我告诉模型用户不会想看这部动漫。y由score和status综合决定，训练集中的score以及status都是为了让我们能够更准确的在未来为用户来推荐他们会想看的动漫。举个简单的例子，如果status=2,score=5，说明用户看完了，但是只打了5分，那我得到的信息是：用户不会喜欢类似的动漫，因此虽然他是看完了，但是我给它的y应该是0而不是1。

&nbsp; &nbsp; &nbsp; &nbsp;关于测试集的标签y，遵循着我想要我推荐的N部动漫尽可能多的被这个用户观看了，因此**在测试集上我关心的是用户究竟看没看我推荐的动漫。**这个时候就不考虑最后的打分影响了，即使某个用户只打了4分，但是他看了，那这个样本的y就应该是1。所以最后测试集上的y的决定标准如下：如果my_watched_episodes > 0，即有观看集数了，则y =1；否则y = 0。


### 3.2.3 用户特征和动漫特征的生成
&nbsp; &nbsp; &nbsp; &nbsp;动漫特征主要作了如下的处理：

- type字段适当扩充，将占据大头的TV动画细分成6类
- 合并了一部分的source字段，最后保留9类；
- 将打分人数scored_by和成员数members合并成一个新字段scored_ratio；
- members的数值区间太大了,简化成6个区间,对应不同的流行度
- 流派标签genre的个数精简到了20个，因为同一部动漫可以对应多个标签，因此统一作了one hot处理

&nbsp; &nbsp; &nbsp; &nbsp;用户特征主要作了如下的处理：

- 根据出生日期，计算用户的年龄age（时间点取为训练集和测试集的划分界限2017-06-30，年龄向下取整）
- 统计每个用户在训练集中各个status的记录数（watching, completed, hold on, dropped）
- 统计每个用户在训练集中“喜欢”了多少动漫（即y = 1的记录数）
- 统计每个用户“喜欢”的动漫中，各个source的比例；各个rating的比例；各个genre的比例
- 统计每个用户的planned to watch的动漫信息中，各个source的比例；各个rating的比例；各个genre的比例


## 3.3 多路召回策略
&nbsp; &nbsp; &nbsp; &nbsp;我的实验里现在一共采用了4种召回策略：item-CF召回、item-related召回、人口信息召回以及item-Similarity召回。一个需要注意的是冷启动问题：**一部分在测试集里出现的动漫，在17年7月之前未上映，因此不会有任何喜欢的记录，属于物品冷启动。同样的，有一部分用户是在17年7月以后才注册的，这部分用户也不会有任何的历史行为记录可用，属于用户冷启动。**这四种召回策略中，item-CF和人口信息召回的都是旧番；item-Similarity召回的是新番；item-related召回的大部分是旧番，少部分是新番。这也是出于对测试集中用户平均观看新番/旧番的比例而定的。在测试集中，每个用户平均观看的新番旧番比例大约是3:7左右。

&nbsp; &nbsp; &nbsp; &nbsp;

### 3.3.1 item-CF召回
&nbsp; &nbsp; &nbsp; &nbsp;用的就是原始经典的基于物品的协同过滤算法，遵循的原理是：和用户历史上感兴趣的物品越相似的物品，越容易出现在用户的推荐列表里。这里我们把item-CF仅仅作为召回算法，因此就变成了：**和用户历史上感兴趣的物品越相似的物品，越可能被用户感兴趣，因而进入召回候选集**。在经过采样后发现，5400名用户在训练集里喜欢过的动漫有接近5000部，但是其中有近50%的动漫只被少于25个人喜欢了，并且平均得分都不高于8分，因此为了简化相似矩阵的运算，我们把这部分动漫去除，只保留2500部动漫进行collaborative filtering。最终召回的时候，参数的选取是K1=K2=10。其中K1是找和用户喜欢的每一部动漫相似度最高的K1部动漫，而K2是留下最后综合评分最高的
K2部动漫作为每个用户的召回结果。现在取的K1=10，K2=20

### 3.3.2 item-related召回
&nbsp; &nbsp; &nbsp; &nbsp;这是根据数据集的特征所做的特殊推荐。anime_clean数据集中有一个字段叫related，里面存储了和这部动漫有所关联的动漫、漫画等。涉及到的关联方式主要有：

- Adaptation：改编，通常都会转到一部漫画，这里我们是做动漫推荐，因此用不上
- Sequel/Prequel:：续集/前传。这一对是一一对应的，如果A是B的Sequel，那B就会是A的Prequel。**动漫A的ova通常会是作为A的Sequel相关出现。**
- Parent story/Side story：父篇/番外。这一对也是一一对应的，如果A是B的Parent story，那B就会是A的Side story。
- 其他种种。。

最终我主要采用的就是Sequel, Prequel, Parent story和Side story这四大类。召回的方法是：根据每个用户历史上喜欢的动漫按照打分从高到低排序，然后依次通过related方式联结到一部新的动漫，如果这部动漫没有被用户看过，那就进入召回列表。最后我选取的K=10。

### 3.3.3 人口信息召回
&nbsp; &nbsp; &nbsp; &nbsp;这部分召回主要就是为了解决用户冷启动的。因为新用户，或者是一部分虽然在训练集里出现过，但是没有留下任何正样本的老用户，没法通过前面两种召回方法进行召回，所以只能通过一些常规方法进行处理。这儿选择的方法是根据用户信息中的性别gender+年龄联合召回。性别分男女还有Non-Binary；年龄按照不同区间，划分成五个区间：19岁以下、19-22岁、22-26岁、26-30岁以及30岁以上。对于测试集中的任意一个用户，根据他的性别还有年龄，找到训练集中和他同性别和年龄区间的人最喜欢的动漫作为召回结果。（如果性别是Non-Binary，则找同年龄段的男女各一半）。最后我选取的K=20。

### 3.3.4 itme-Similarity召回
&nbsp; &nbsp; &nbsp; &nbsp;这部分召回主要是为了解决物品冷启动的。因为新番是肯定不会在训练集里出现的，也就不存在被“喜欢”一说，从而无法通过前面三种方法召回。这里采用的方法是计算新番和每一部老番的余弦相似度，然后对于用户在训练集里喜欢的动漫，找到和它们最相似的K部新番进行召回。如果不巧这个用户还是个新用户，那就找和人口信息召回的动漫最相似的K部新番。余弦相似度的计算和前面的item-CF的算法类似，**首先给每一部动漫，确定一系列特征，把动漫“向量化”。**然后把每一个特征看成是item-CF里的一个用户，接着只要用相同的倒排表、共现矩阵的算法就可以算出新番和老番之间的cosine distance了。最后我给每部动漫选择的特征是20维的genre特征 + 2维的稀有rating特征 + 9维的source特征，一共31维。这里的K我取得是20，因为最后期望推荐的新番数大概是10部左右吧。

## 3.4 排序算法
&nbsp; &nbsp; &nbsp; &nbsp;传统机器学习应用于排序阶段的，最常见的就是LR和GBDT，还有GBDT+LR的组合。不过我一开始用sklearn的GBDT训练了一个小时也没出结果。。。后面我就换成了XGBoost，结果几分钟就训练好了。。暂时还没有试其他的算法

## 3.5 评估指标
&nbsp; &nbsp; &nbsp; &nbsp;下面是四种召回算法在测试集上的各指标统计，K值暂时就取了10和20。

|K|单路召回算法 | 召回率 | 精准率 | 命中率 | 覆盖率(item占比) | 覆盖率(熵) | 覆盖率(type的熵) | 冲撞率(zero) | 冲撞率(plan) |
| ----- | :-----   | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 10 | item-CF | 2.093% | 10.091% | 41.798% | 10.033% | 6.7965 | 0.4407 | 1.659% | 1.531% |
|    | item-related |1.466% | 7.247% | 35.953%| 22.615%| 9.0377 |2.0616| 0.809% | 0.415% |
|    | gender+age | 2.036%  | 9.177%  | 41.999%  |3.824%  | 6.0121  |1.0060  |1.107%  | 1.160% |
|    | new   | 2.445% |  11.000%|  47.561% | 4.709%|   6.5822 | 1.7849 |   0.000% |  0.000%| 
| 20 | item-CF | 3.709% | 8.972% | 54.252% |15.102% |7.5552 |0.6080 |2.991% | 2.818%|
|    | item-related | 2.656% |  6.770% |  48.609%|  27.070% | 9.3214|  2.0684 | 1.510% |  0.757%| 
|    | gender+age |  3.641% | 8.230%  |52.358% |5.324% | 6.8802| 0.9516 |2.188% | 2.239%|
|    | new  |4.208% | 9.498% | 57.517% |5.624% | 6.9978 |1.8106| 0.000% | 0.000%|

其中冲撞率是我自己定义的，计算方法和召回率一样，就是把测试集的位置换成train_zero或者train_planned_to_watch。
大体上来看，可以得出下面的粗略结论：

- 人口统计召回和新番召回的覆盖率非常低，这两种方法召回的动漫相对是比较集中，属于比较热门的动漫。
- 从Recall和Precision来看，新番召回最高，而协同过滤和人口统计召回差不多
- 通过related字段进行召回的结果，无论是recall,precision还是命中率表现都是最差的，但是它召回的动漫分布最丰富，多样性最好；同时，它召回的动漫和用户缺失打分的数据相撞的比例也是最低的。（新番召回不会相撞因此不算在内）
- 从统计结果来看，冲撞率和召回率比较接近，后续应该要考虑把这部分缺失打分的数据给利用上。

&nbsp; &nbsp; &nbsp; &nbsp;根据上面四种算法各自的表现，结合测试集中用户观看旧番/新番的比例，对四种召回策略进行组合，然后利用训练好的XGBoost模型进行排序预测打分，取每个用户评分最高的10部新番和20部旧番作为最终推荐结果。模型的参数还没有调优，只是初始化取了一些参数而已。最后的效果如下：

|召回组合,是否排序| 召回率 | 精准率 | 命中率| 覆盖率(item占比) | 覆盖率(熵) | 覆盖率(type的熵) | 冲撞率(zero) | 冲撞率(plan) |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 不排序，全量推荐|8.430% | 11.292% |84.119% |34.883% |8.9747 |1.5907 |4.869% | 4.360%|
| 排序，取新番前10旧番前20| 7.379% |11.068% |77.952% |17.921% |7.8151 |1.3029 |2.625% | 2.389%|

&nbsp; &nbsp; &nbsp; &nbsp;可以看到：在精准率基本保持的情况下，召回率、命中率、覆盖率等都相对单个算法有大幅度的提升，不过冲撞率也有小幅提升。下面的图是模型给出的特征重要性的前20名：
![](C:\Users\tqd95\Desktop\graduation_thesis\result\feature_importance.png)
## 3.6 现有的问题和改进方向

- 我的考虑是从召回算法入手，因为训练集挺大的，采样后还有90w行左右，排序算法估计跑不动。。或者考虑继续精简数据集到一个更小的量级吗
- 召回冲撞问题。因为我在训练集里删掉了status=123 & score=0的记录，以及status=6 & score=0的记录。而在后面召回过滤的时候，只考虑了真实训练集里(打分非零)用户看过的动漫，而没有考虑到这部分看了/打算看但是没有评分的记录，因此可能会出现在测试集中，对这个用户推荐的动漫，其实在之前这个用户已经看了/看完了/搁置/打算看，只不过评分为0而已。
- 接着上面的，对于原始训练集中status=123 &score=0的记录能不能**先通过某种方法预测打分，然后再把这一部分的信息加到训练集里去？因为**这是一个评分预测问题，可选的经典方法有KNN、矩阵分解(FunkSVD,BiasSVD,SVD++等等).
- 但是找不到什么创新的东西。。。或许可以试试把status，还有watched episodes等作为权重加到损失函数里？这部分特征我现在没用过
- 文献综述里，HybridSVD和internrelationship的论文感觉相对比较容易，另外的基本都是基于深度学习或是别的而且主要针对排序算法，我也没找到什么比较好的想法。
- new的推荐，我现在的做法是




【参考文献】

[1] Resnick P,Iacovou N, Suchak M, et al. GroupLens: an open architecture for collaborativefiltering of netnews[C] Proceedings of the 1994 ACM Conference on ComputerSupported Cooperative Work, Oct 22-26, 1994. New York, NY, USA: ACM, 1994:175-186.

[2] Resnick P, Varian H R. Recommender systems[J].Communications of the ACM, 1997, 40(3): 56-58.

[3] Linden G, Smith B, York J. Amazon. com recommendations: Item-to-item collaborative filtering[J]. Internet Computing, IEEE, 2003, 7(1): 76-80. 

[4] Adomavicius G, Tuzhilin A. Toward the nextgeneration of recommender systems: a survey of the state-of-the-art and possibleextensions[J]. IEEE Transactions on Knowledge and Data Engineering, 2005,17(6): 734-749.

[5] Koren Y , Bell R , Volinsky C . Matrix Factorization Techniques for Recommender Systems[J]. Computer, 2009, 42(8):30-37.

[6] Rendle S, Freudenthaler C, Gantner Z, et al. BPR: Bayesian personalized ranking from implicit feedback[C]// Conference on Uncertainty in Artificial Intelligence. 2009.

[7] Koren Y . Factorization meets the neighborhood: A multifaceted collaborative filtering model[C]// Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Las Vegas, Nevada, USA, August 24-27, 2008. ACM, 2008.

[8] Rendle S . Factorization Machines[C]// IEEE International Conference on Data Mining. IEEE, 2011.

[9] Cheng H T, Koc L, Harmsen J, et al. Wide & Deep Learning for Recommender Systems[J]. 2016.

[10] Guo H , Tang R , Ye Y , et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction[J]. 2017.

[11] Wang X , He X , Feng F , et al. TEM: Tree-enhanced Embedding Model for Explainable Recommendation[C]// the 2018 World Wide Web Conference. 2018.

[12] Xue H J , Dai X , Zhang J , et al. Deep Matrix Factorization Models for Recommender Systems[C]// Twenty-Sixth International Joint Conference on Artificial Intelligence. AAAI Press, 2017.

[13] Frolov E , Oseledets I . HybridSVD: When Collaborative Information is Not Enough[J]. 2018.

[14] Zhang Z P, Kudo Y, Murai T, et al. Addressing Complete New Item Cold-Start Recommendation: A Niche Item-Based Collaborative Filtering via Interrelationship Mining[J]. Applied Sciences, 2019, 9(9): 1894.