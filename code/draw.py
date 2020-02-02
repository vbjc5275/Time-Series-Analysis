# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 21:06:18 2020

@author: Jerry
"""
#%%
#basic
import pandas as pd
import numpy as np
#draw
import matplotlib.pyplot as plt
#PAA
from tslearn.piecewise import PiecewiseAggregateApproximation
#關掉warning
import warnings
warnings.filterwarnings('ignore')
#%%
normal = pd.read_csv(r"circular_move(processed data).csv")
normal = normal[normal["circle_size"]==11]
normal.columns
columns = ['P mech/X','P mech/Y', 'P mech/Z','I actual/X','I actual/Y', 'I actual/Z']

plt.figure(figsize=(35,30))
for i,col in enumerate(columns):
    plt.subplot(2,3,i+1)
    plt.plot(normal[col].values)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.title(col,fontsize=18)
    plt.xlabel("Sample",fontsize=17,fontweight='bold')
    plt.tick_params(axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False)
   
    
normal = normal[normal.th==1]
plt.figure(figsize=(35,30))
for i,col in enumerate(columns):
    plt.subplot(2,3,i+1)
    plt.plot(normal[col].values)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.title(col,fontsize=18)
    plt.xlabel("Sample",fontsize=17,fontweight='bold')
    plt.tick_params(axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False)
 

#PAA降維
plt.figure(figsize=(35,30))
n_sigments = 500
for i,col in enumerate(columns):
    tmp = get_mean(normal[col].values.reshape(1,-1),n_sigments)
    plt.subplot(2,3,i+1)
    plt.plot(tmp[0])
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.title(col,fontsize=18)
    plt.xlabel("Sample",fontsize=17,fontweight='bold')
    plt.tick_params(axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False)

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


#plt.savefig(r"D:\大專生科技部\image\UX300 circle_size11\each circle.png",dpi=600)
