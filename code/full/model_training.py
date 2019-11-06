# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 09:58:46 2019
训练XGBoost模型
@author: tqd95
"""
import pandas as pd
import xgboost as xgb
import time
import pickle
from tqdm import tqdm

# 模型参数
xgb_params_init = {'objective':'binary:logistic',
                  'eta':0.5,
                  'max_depth':5,
                  'min_child_weight':100,
                  'gamma':1,
                  'subsample':0.8,
                  'colsample_bytree':0.8,
                  'n_jobs':-1,
                  'alpha':0,
                  'lambda':1,
                  'seed':1001
                  }
xgb_params = {'objective':'binary:logistic',
              'eta':0.5,
              'max_depth':5,
              'min_child_weight':100,
              'gamma':1,
              'subsample':0.8,
              'colsample_bytree':0.8,
              'n_jobs':-1,
              'alpha':0,
              'lambda':1,
              'seed':1001,
              'process_type':'update',
              'updater':'refresh',
              'refresh_leaf':True
              }

test_fn = pd.read_pickle(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\test_fn.pkl')
xg_test = xgb.DMatrix(test_fn.values, nthread=-1)
del test_fn


# 增量训练Xgboost模型
def incremental_training():
    chunkSize = 500000  # 一次读入50w行数据
    reader = pd.read_csv(r'C:\Users\tqd95\Desktop\graduation_thesis\dataset\full\train_fn.csv',
                             chunksize=chunkSize)
    rounds = 1 # 训练轮数
    start = time.time()
    model = None
    print('training begins.')
    with tqdm(total=36) as pbar:
        for df in reader:
            y_train = df['label'].values
            x_train = df.drop('label',axis=1)
            x_train = x_train.drop('Unnamed: 0', axis=1).values
            xg_train = xgb.DMatrix(x_train, label = y_train)
            # 判断本地模型是否存在, 是就读取模型, 作为本轮增量训练的参数`xgb_model`传入
            # 默认工作路径为'C:\Users\tqd95'
# =============================================================================
#             if not os.path.exists('full_xgboost.pkl'):
#                 old_model = None
#             else:
#                 with open('full_xgboost.pkl', 'rb') as fr:
#                     old_model = pickle.load(fr)
# =============================================================================
            # 增量训练模型
            if not model:
                model = xgb.train(xgb_params_init, dtrain = xg_train, num_boost_round = 50)
            else:
                model = xgb.train(xgb_params, dtrain = xg_train, num_boost_round = 50,
                                  xgb_model = model)
            
            # 每轮的模型预测一次结果,写入文件中
            # xgb的原生接口没有predict_proba方法,只能用predict方法
            #y_pred = model.predict_proba(test_fn)[:,1]
            y_pred = model.predict(xg_test)
            predict_res = pd.DataFrame(y_pred).rename(columns = {0:'pred_score'})
            file_path = 'C:\\Users\\tqd95\\Desktop\\graduation_thesis\\result\\incremental_pred\\pred_' + str(rounds) + '.csv'
            predict_res.to_csv(file_path, index = False)
            rounds += 1
            pbar.update(1)
            
    # 保存模型到本地pickle文件
    with open('full_xgboost.pkl','wb') as fw:
        pickle.dump(model, fw)
    end = time.time()
    print('training completed!')
    print('total time used:', (end-start)/60, 'minutes')
    return

## 开始训练！！
incremental_training()


