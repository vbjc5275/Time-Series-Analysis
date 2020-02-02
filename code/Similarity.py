# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:22:27 2020

@author: Jerry
"""
#%%
#basic
import numpy as np
import pandas as pd
from scipy import stats
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity

#%%
path = r"D:\大專生科技部\dataset\UX300\size11 (30,6307)"
normal = pd.read_csv(path+f"\\normal\\I actualX.csv")

abnormal = pd.read_csv(path+f"\\abnormal\\I actualX_v1.csv")
abnormal2 = pd.read_csv(path+f"\\abnormal\\I actualX_v2.csv")
abnormal3 = pd.read_csv(path+f"\\abnormal\\I actualX_v3.csv")
abnormal4 = pd.read_csv(path+f"\\abnormal\\I actualX_v4.csv")
abnormal = np.vstack((abnormal.iloc[0,:].values,abnormal2.iloc[0,:].values,
                      abnormal3.iloc[0,:].values,abnormal4.iloc[0,:].values))
del abnormal2,abnormal3,abnormal4

X = np.concatenate((normal,abnormal))
y = [0 for _ in range(len(normal))]  
y.extend([1 for _ in range(len(abnormal))])
y = np.array(y)

#正常、異常樣本四 cosine
test_X = np.vstack((X[0,:],X[-1,:]))
cos_sim = cosine_similarity(X,test_X)
d_sim = dtw_dis(X,test_X)
edu = edu_dis(X,test_X)

def knn_with_eculid(X,y,test_X,k):
    pred = []
    sim = cosine_similarity(X,test_X)
    sort = np.sort(sim,axis=0)[-k:,:][::-1]
    for col in range(sort.shape[1]):
        #值可能1對多 因此攤開取前k個
        similar_index = [[i[0] for i in np.argwhere(sim[:,col]==sort[i,col])] for i in range(k)]
        similar_index = [i for item in similar_index for i in item][:k]
        pred.append(stats.mode(y[similar_index])[0][0])
    return pred

def knn_with_cosine_similarity(X,y,test_X,k):
    pred = []
    sim = cosine_similarity(X,test_X)
    sort = np.sort(sim,axis=0)[-k:,:][::-1]
    for col in range(sort.shape[1]):
        #值可能1對多 因此攤開取前k個
        similar_index = [[i[0] for i in np.argwhere(sim[:,col]==sort[i,col])] for i in range(k)]
        similar_index = [i for item in similar_index for i in item][:k]
        pred.append(stats.mode(y[similar_index])[0][0])
    return pred

def knn_with_dtw_dis(X,y,test_X,k):
    pred = []
    sim = dtw_dis(X,test_X)
    sort = np.sort(sim,axis=0)[-k:,:]
    for col in range(sort.shape[1]):
        #值可能1對多 因此攤開取前k個
        shorter_index = [[i[0] for i in np.argwhere(sim[:,col]==sort[i,col])] for i in range(k)]
        shorter_index = [i for item in shorter_index for i in item][:k]
        pred.append(stats.mode(y[shorter_index])[0][0])
    return pred

def dtw_dis(arr1,arr2):
    dis = np.zeros((arr1.shape[0],arr2.shape[0]))
    length = len(arr2)
    for i in range(length):
        fun = lambda x: dtw.distance_fast(x,arr2[i])
        dis[:,i] = np.apply_along_axis(fun,1,arr1)
    return dis
    

def edu_dis(arr1,arr2):
    dis = np.zeros((arr1.shape[0],arr2.shape[0]))
    length = len(arr2)
    for i in range(length):
        fun = lambda x: np.sqrt(sum(pow(x-arr2[i],2)))
        dis[:,i] = np.apply_along_axis(fun,1,arr1)
    return dis
    
def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a 
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim