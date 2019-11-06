# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 09:48:31 2019

@author: tqd95
"""

import numpy as np
import time

from sklearn.decomposition import NMF

M = np.array([[1,0,1,1,0],[1,1,0,1,0],[1,0,0,1,0]])
nmf = NMF(n_components=2)

P = nmf.fit_transform(M)
Q = nmf.components_

tom = np.array([[1,0,0,1,1]])
nmf.transform(tom)

P1 = nmf.transform(tom)




M1 = np.floor(np.random.rand(5000,5000)*10)
start = time.time()
nmf = NMF(n_components=10)
P1 = nmf.fit_transform(M1)
end = time.time()
print('time used:', end-start, 's')
Q1 = nmf.components_
