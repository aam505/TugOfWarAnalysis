import matplotlib.pyplot as plt
import pandas as pd     
import numpy as np
import os
import glob
import seaborn as sns
from sklearn import preprocessing
import matplotlib.gridspec as gridspec
from collections import OrderedDict
from collections import deque
#################################################################################3
# Swap function 

width=0.4
#################################################################################3

def load_csvs(path):
    p_dict={}
    """Returns a list with all files with the word/extension in it"""
    word="csv"
    for f in glob.glob(path):
        if word in f:
            df=pd.read_csv(f).drop(columns=['Unnamed: 0'])
            p_dict[f[44:-4]] =  df

    return p_dict

pull_data=load_csvs("D:\\Master\\Thesis\\Results Scripts\\Repo\\pulls\\**")

pids = list(pull_data.keys()) 

#data.to_csv('D:\\Master\\Thesis\\Results Scripts\\Repo\\data.csv')

#normalize between 0,1  
min_max_scaler = preprocessing.MinMaxScaler()

#Make pull by condition and pull by force tables for all participants
for i in range (0,len(pids)):
     pull_data[pids[i]]['ForceNormalized'] = min_max_scaler.fit_transform(pd.DataFrame( pull_data[pids[i]].iloc[:,-1]))


#aggregate
data=pull_data[pids[0]].copy()
for i in range (1,len(pids)):
    data=data.append(pull_data[pids[i]], ignore_index = True)

first = data.groupby('Pid')[['Pid', 'Force']].first().reset_index()
data['NForce'] = data[['Pid', 'Force']].apply(lambda x: x['Force'] / first[first['Pid'] == x['Pid']]['Force'])

print(first)
def normalize(x):
    first = x[x['Trial'] == 1]['Force'].iloc[0]
    print(x['Force'], x['Force']/ first)
    return  x['Force'] / first
normed = data.groupby('Pid').apply(normalize).reset_index()
print(normed)
data['NForce'] = normed['Force']
print(data)

#data.to_csv('D:\\Master\\Thesis\\Results Scripts\\Repo\\data.csv')

sns.catplot(x="Condition", y="ForceNormalized", hue="Gender", kind="bar", data=data);

g = sns.catplot(x="Condition", y="Force", kind="violin", inner=None, data=data)
sns.swarmplot(x="Condition", y="Force", color="k", size=5, data=data, ax=g.ax);

def heatmaps():
    sns.heatmap(results_by_index_challenge.iloc[:,1:], annot=True)    
    sns.heatmap(results_by_index_ppul.iloc[:,1:], annot=True)    
    #heatmap for results by condition challenge
    #heatmap for results by condition challenge