# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:42:38 2020

@author: Jerry
"""

import time
import pandas as pd
import matplotlib.pyplot as plt

path = r"D:\大專生科技部\dataset\UX300\circular_move 0611.csv"
circular_move = pd.read_csv(path,sep=";")
circular_move.columns = [c.strip() for c in circular_move.columns]
circular_move = circular_move[[c for c in circular_move.columns if c !=""]]
circular_move.columns
circular_move = circular_move.astype(float)
circular_move['Block no./CH_NC'].unique()

start_list = [21, 30, 39, 48, 57, 66, 75, 84, 93, 102, 111]
df = circular_move[circular_move['Block no./CH_NC'].isin(start_list)].reset_index(drop=True)

#circle_size
size = 11
for no in start_list:
    df.loc[df['Block no./CH_NC'] == no,"circle_size"] = size
    size -= 1
df["circle_size"].value_counts()

#th
#執行一圈的過程中時間為連續 因此時間間隔如果>3ms 那就是換下一圈了。
df["th"] = 0
df["diff"] = df['Time[us]'].diff()
all_th_start_index = list(df[df["diff"]>3000].index) #所有圓弧的開頭索引
all_th_start_index.insert(0,0)
all_th_start_index.append(100000000)


ths = [i for i in range(1,31)]
th_cycle = [th+1 for th in range(30)]
i = 0
for cur in range(len(all_th_start_index)-1):
    mask = (all_th_start_index[cur]<=df.index) & (df.index<all_th_start_index[cur+1])
    df.loc[mask,"th"] = th_cycle[i]
    i = i+1 if i<29 else 0

df.th.value_counts()
df.drop(["diff"],axis=1,inplace=True)
df.to_csv("circular_move(processed data).csv",index=False)

#%%
#對其中一個尺寸圓，每一cycle作為row
'''
df = df[df.circle_size==11]
columns = list(df.columns[2:-2])
for col in columns:
    tmp = df[[col,"th"]].copy()
    d = {}
    for th in tmp.th.unique():
        d[th] = tmp.loc[tmp.th==th,col].values[:6307].tolist()
    col = col.replace("/","")
    pd.DataFrame(d).T.to_csv(f"../dataset/UX300/size11 (30,6307)/{col}.csv",index=False)
'''

#%%
radius = [0.5,5,10,15,20,25,30,35,40,45,50]
col = "I actual/X"
col = 'P mech/X'
col = 'P mech/Y'
plt.figure(figsize=(20,10))
plt.plot(df[(df.circle_size==11)&(df.th==1)][col].values)
plt.plot(df[df.circle_size==11][col].values)
df.th.value_counts()

#%%
#繪圖
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 20000

columns = list(circular_move.columns[2:])

df["Sample"] = [i for i in range(df.shape[0])]

for column in columns: 
    plt.figure(figsize=(25,20))
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel("Sample",fontsize=25,fontweight='bold')
    plt.ylabel(column,fontsize=25,fontweight='bold')
    plt.legend(loc="upper left",fontsize=15) # loc="upper right"
    sns.lineplot(x="Sample",y=column,data=df[df.circle_size==11])
    imgpath = r"D:\大專生科技部\image"
    column = column.replace("/","")
    plt.savefig(f"{imgpath}\\{column}.png")
    