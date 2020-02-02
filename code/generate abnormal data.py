# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 10:14:45 2020

@author: Jerry
"""
#basic
import pandas as pd
import numpy as np
#關掉warning
import warnings
warnings.filterwarnings('ignore')
#draw
import matplotlib.pyplot as plt


normal = pd.read_csv(r"../dataset/UX300/size11 (30,6307)/normal/I actualX.csv")
plt.plot(normal.iloc[0,:].values)

imgpath = r"..\image\UX300 circle_size11\abnormal"
abnormal_path = r"../dataset\UX300\size11 (30,6307)\abnormal"
#異常類別-1
abnormal = normal.iloc[0,:].values
mu, sigma = 0, 0.07
start, end = (2500,3000)
abnormal = make_abnormal_data_1(normal.iloc[0,:].values,(start, end),mu,sigma)

#draw
plt.plot(abnormal)
plt.scatter(start,0,c="r",s=50)
plt.scatter(end,0,c="r",s=50)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlabel("Sample",fontsize=17,fontweight='bold')
plt.savefig(f"{imgpath}\I actualX_v1")

#to_csv
saved_data = pd.DataFrame(np.concatenate((abnormal,abnormal,abnormal),axis=0).reshape(3,len(abnormal)))
saved_data.to_csv(f'{abnormal_path}\I actualX_v1.csv',index=False)


#異常類別-2
abnormal = normal.iloc[0,:].values
mu, sigma = 0, 0.0005
start, end = (4000,4300)
abnormal = make_abnormal_data_2(normal.iloc[0,:].values,(start,end),mu,sigma)

#draw
plt.plot(abnormal)
plt.scatter(start,0,c="r",s=50)
plt.scatter(end,0,c="r",s=50)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlabel("Sample",fontsize=17,fontweight='bold')
plt.savefig(f"{imgpath}\I actualX_v2")

#to_csv
saved_data = pd.DataFrame(np.concatenate((abnormal,abnormal,abnormal),axis=0).reshape(3,len(abnormal)))
saved_data.to_csv(f'{abnormal_path}\I actualX_v2.csv',index=False)


#異常類別-3
abnormal = normal.iloc[0,:].values
mu, sigma = 0, 0.0005
start, end = (1000,1300)
abnormal = make_abnormal_data_3(normal.iloc[0,:].values,(start,end),mu,sigma)

#draw
plt.plot(abnormal)
plt.scatter(start,0,c="r",s=50)
plt.scatter(end,0,c="r",s=50)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlabel("Sample",fontsize=17,fontweight='bold')
plt.savefig(f"{imgpath}\I actualX_v3")

#to_csv
saved_data = pd.DataFrame(np.concatenate((abnormal,abnormal,abnormal),axis=0).reshape(3,len(abnormal)))
saved_data.to_csv(f'{abnormal_path}\I actualX_v3.csv',index=False)

#異常類別-4 位移
abnormal = np.concatenate((normal.iloc[0,:].values,normal.iloc[1,:].values))
abnormal = pd.Series(abnormal,index=pd.date_range('1/1/2000', periods=abnormal.shape[0], freq='3ms'))

start, end = 5800, 6307
#draw
plt.plot(abnormal[start:start+end].values)
plt.scatter(0,0,c="r",s=50)
plt.scatter(end-start,0,c="r",s=50)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlabel("Sample",fontsize=17,fontweight='bold')
plt.savefig(f"{imgpath}\I actualX_v4")

abnormal = abnormal[start:start+end].reset_index(drop=True)
saved_data = pd.DataFrame(np.concatenate((abnormal,abnormal,abnormal),axis=0).reshape(3,len(abnormal)))
saved_data.to_csv(f'{abnormal_path}\I actualX_v4.csv',index=False)

#常態亂數
mu, sigma = 0, 0.03
y = np.random.normal(mu, sigma)
# 4x4维服从正態分布的数组
y = [np.random.normal(mu, sigma) for i in range(500)]

def make_abnormal_data_1(normal,interval,mu,sigma):
    abnormal = normal.copy()
    start, end = interval
    y = [np.random.normal(mu, sigma) for i in range(500)]
    abnormal[start:end] += y
    return abnormal


def make_abnormal_data_2(normal,interval,mu,sigma):
    """常態亂數遞增"""
    abnormal = normal.copy()
    start, end = interval
    ran = []
    cum = 0
    for i in range(end-start):
        cum += abs(np.random.normal(mu, sigma))
        ran.append(cum)
    abnormal[start:end] += ran
    return abnormal


def make_abnormal_data_3(normal,interval,mu,sigma):
    """常態亂數遞減"""
    abnormal = normal.copy()
    start, end = interval
    ran = []
    cum = 0
    for i in range(end-start):
        cum += abs(np.random.normal(mu, sigma))
        ran.append(cum)
    abnormal[start:end] -= ran
    return abnormal   

