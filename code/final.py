# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 08:30:05 2020

@author: Jerry
"""
#basic
import pandas as pd
import numpy as np
from collections import defaultdict
import itertools
from scipy.stats import norm
#sklearn
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import confusion_matrix,roc_curve, auc,accuracy_score
from sklearn.model_selection import cross_validate, cross_val_predict,train_test_split
#sax
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize
from saxpy.paa import paa
#
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation, \
    OneD_SymbolicAggregateApproximation
#關掉warning
import warnings
warnings.filterwarnings('ignore')
#draw
import matplotlib.pyplot as plt
import time

def simple_predict(models,train_X,test_X,train_y,test_y):
    d = {}
    for model in models:
        algorithm = str(model)[:str(model).find("(")]
        d[algorithm] = {}
        clf = model.fit(train_X,train_y)
        y_pred =  clf.predict(test_X)
        d[algorithm] = accuracy_score(test_y, y_pred) 
    return d

def to_list_string(arr,asize):
    paa_string = np.apply_along_axis(lambda x:ts_to_string(x,cuts_for_asize(asize)), 1, arr)
    paa_string = paa_string.reshape(paa_string.shape[0],1)
    paa_list_string = np.apply_along_axis(lambda s:list(s[0]), 1, paa_string )
    return paa_list_string

def alpha_to_number(arr):
    alpha = {"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,"h":8}
    for ch in alpha:
        arr  = np.where(arr==ch, alpha[ch], arr) 
    return arr

def get_mean(arr,n_sigments):
    n_row = arr.shape[0]
    sigment_len = arr.shape[1]//n_sigments
    
    if sigment_len*n_sigments != arr.shape[0]:
        paa = PiecewiseAggregateApproximation(n_segments=n_sigments)
        new_arr = paa.fit_transform(arr)
        return new_arr.reshape(arr.shape[0],n_sigments)
    
    new_arr = np.zeros((n_row,n_sigments))
    for i in range(n_row):
        new_arr[i] = np.apply_along_axis(np.mean,1,arr[i].reshape(n_sigments,sigment_len))
    return new_arr

def join_same_class(X,y):
    """same class combine to single list"""
    d = []
    for c in np.unique(y):
        index = [i[0] for i in np.argwhere(y==c)]
        d.append(" ".join(X[index]))
    return d

def get_sax_weight(X,y,asize):
    X = np.apply_along_axis(lambda x:ts_to_string(x,cuts_for_asize(asize)), 1, X)
    word_weight = manyseries_to_wordbag(join_same_class(X,y)).reset_index()
    word_weight.columns = ["alpha","normal","abnormal"] 
    return pd.DataFrame(X,columns=["alpha"]).merge(word_weight[["alpha","normal"]],on="alpha",how="left")
          
def manyseries_to_wordbag(d):
    vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
    tfidf=transformer.fit_transform(vectorizer.fit_transform(d))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
    weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    return pd.DataFrame(weight,columns =word).T
    
def to_paa_list_string(arr,asize):
    paa_string = np.apply_along_axis(lambda x:ts_to_string(x,cuts_for_asize(asize)), 1, arr)
    paa_string = paa_string.reshape(paa_string.shape[0],1)
    paa_list_string = np.apply_along_axis(lambda s:list(s[0]), 1, paa_string )
    alpha = {"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,"h":8,"i":9,"j":10}
    for ch in alpha:
        paa_list_string  = np.where(paa_list_string==ch, alpha[ch], paa_list_string) 
    return paa_list_string

def get_score_with_diff_asize(X_train, X_test, y_train, y_test):
    #model = GradientBoostingClassifier()
    model = DecisionTreeClassifier()
    #model = KNeighborsClassifier(n_neighbors=3)
   
    d  = {}
    for asize in [3,4,5,6,7,8]:
        y_pred = model.fit(to_paa_list_string(X_train,asize),y_train).predict(to_paa_list_string(X_test,asize))
        d[asize] = round(accuracy_score(y_test, y_pred),4)
    return d

def get_score_with_diff_asize_by_weight(X_train, X_test, y_train, y_test):
    #model = DecisionTreeClassifier()
    model = SAXVSM(window_size=34, sublinear_tf=False, use_idf=False)
    d = {}
    for asize in [3,4,5,6,7,8]:
        X_train_weight = get_sax_weight(X_train,y_train,asize=asize)
        X_test_weight = get_sax_weight(X_test,y_test,asize=asize)
       
        #*weight
        x_train = (X_train.T*X_train_weight.normal.values).T
        x_test = (X_test.T*X_test_weight.normal.values).T
        y_pred = model.fit(x_train,y_train).predict(x_test)
        d[asize] = round(accuracy_score(y_test, y_pred),4)
    return d

path = r"D:\大專生科技部\dataset"

dataset = ["I actualX 1500.csv","I actualX abnormal.csv",
           "I actualY 1500.csv","I actualY abnormal.csv",
           "I actualZ 1500.csv","I actualZ abnormal.csv"]
i = 4
normal = pd.read_csv(path+f"\\{dataset[i]}", header=None)
abnormal = pd.read_csv(path+f"\\{dataset[i+1]}",header=None)
X = np.concatenate((normal,abnormal))
y = [0 for _ in range(1500)]  
y.extend([1 for _ in range(50)])
y = np.array(y)

#拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
  
#SAX-1 paa
n_sigments = 10
X_train = get_mean(X_train,n_sigments)
X_test = get_mean(X_test,n_sigments)

############### consine
import heapq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DistributedCosineKnn:
	def __init__(self, k=3):
		self.k=k

	def fit(self, input_data, n_bucket=1):
	    idxs=[]
	    dists=[]
	    buckets = np.array_split(input_data,n_bucket)
	    for b in range(n_bucket):
	    	cosim = cosine_similarity(buckets[b], input_data)
	    	idx0=[(heapq.nlargest((self.k+1), range(len(i)), i.take)) for i in cosim]
	    	idxs.extend(idx0)
	    	dists.extend([cosim[i][idx0[i]] for i in range(len(cosim))])
	    return np.array(idxs),np.array(dists)
    
cos_knn = sims.DistributedCosineKnn(k=3)
indices, distances = cos_knn.fit(input_data=X, n_bucket=7)

#########
#score_di = get_score_with_diff_asize(X_train, X_test, y_train, y_test)

score = get_score_with_diff_asize_by_weight(X_train, X_test, y_train, y_test)


#get weight
X_train_weight = get_sax_weight(X_train,y_train,asize=8)
X_test_weight = get_sax_weight(X_test,y_test,asize=8)

#*weight
X_train = (X_train.T*X_train_weight.normal.values).T
X_test = (X_test.T*X_test_weight.normal.values).T


#擬和
neigh_model = KNeighborsClassifier(n_neighbors=3)
RF_model = RandomForestClassifier(random_state = 0)
xgb_model = XGBClassifier(random_state=0)
svc_model = SVC(gamma='auto')
models = [RF_model,xgb_model,svc_model,neigh_model]
score = simple_predict(models, X_train, X_test, y_train, y_test)

# =============================================================================
# public data validation
# =============================================================================
#擬和
neigh_model = KNeighborsClassifier(n_neighbors=3)
RF_model = RandomForestClassifier(random_state = 0)
xgb_model = XGBClassifier(random_state=0)
svc_model = SVC(gamma='auto')
models = [RF_model,xgb_model,svc_model,neigh_model]

def open_wafer(train=True):
    if train:
        path = r"D:\大專生科技部\dataset\Univariate_arff\Wafer\Wafer_TRAIN.txt"
    else:
        path = r"D:\大專生科技部\dataset\Univariate_arff\Wafer\Wafer_TEST.txt"
    
    with open(path,'r') as fp:
         all_lines = fp.readlines()
        
    all_lines = [[float(num) for num in line.split(" ") if num!=""] for line in all_lines]
    X = np.array(all_lines)
    y = X[:,0]
    X = X[:,1:] 
    return (X,y)

X_train,y_train = open_wafer()
X_test, y_test = open_wafer(train=False)

#SAX-1 paa
n_sigments = 20
X_train = get_mean(X_train,n_sigments)
X_test = get_mean(X_test,n_sigments)

score_di = get_score_with_diff_asize(X_train, X_test, y_train, y_test)
score = get_score_with_diff_asize_by_weight(X_train, X_test, y_train, y_test)
# =============================================================================
# 
# =============================================================================
