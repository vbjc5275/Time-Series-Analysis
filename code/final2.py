# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 20:17:08 2020

@author: Jerry
"""
#%%
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
import seaborn as sns
import time
#%%
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
    model = XGBClassifier()
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
#%%
path = r"D:\大專生科技部\dataset\UX300\size11 (30,6307)"
normal = pd.read_csv(path+f"\\normal\\I actualX.csv")
abnormal = pd.read_csv(path+f"\\abnormal\\I actualX_v1.csv")

#draw
sns.lineplot(x=[i for i in range(normal.shape[1])], y=normal.iloc[0,:].values)
sns.lineplot(x=[i for i in range(abnormal.shape[1])], y=abnormal.iloc[0,:].values)

#增加樣本
normal = normal.append(normal).reset_index(drop=True)
abnormal = abnormal.append(abnormal).reset_index(drop=True)

X = np.concatenate((normal,abnormal))
y = [0 for _ in range(len(normal))]  
y.extend([1 for _ in range(len(abnormal))])
y = np.array(y)

#%%
#拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
models = [XGBClassifier(),KNeighborsClassifier(),]
a = simple_predict(models,X_train,X_test,y_train,y_test)



#SAX-1 paa
n_sigments = 1000
X_train = get_mean(X_train,n_sigments)
X_test = get_mean(X_test,n_sigments)

score_di = get_score_with_diff_asize(X_train, X_test, y_train, y_test)
score = get_score_with_diff_asize(X_train, X_test, y_train, y_test)
