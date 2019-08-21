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

first = data.groupby('Pid')['Force'].first()
def norm_force(x):
    return x['Force'] / first.loc[x['Pid']]
data['NForce'] = data[['Pid', 'Force']].apply(norm_force, axis=1)
print(data)

#data.to_csv('D:\\Master\\Thesis\\Results Scripts\\Repo\\data.csv')

sns.catplot(x="Trial", y="NForce", hue="Gender", kind="bar", data=data);

sns.catplot(x="Condition", y="Force", hue="Gender", kind="bar", data=data);

g = sns.catplot(x="Condition", y="Force", kind="violin", inner=None, data=data)
sns.swarmplot(x="Condition", y="Force", color="k", size=5, data=data, ax=g.ax);


def heatmap_ppull_idx():
    srted=data.sort_values("Trial",0)
    xlabels=srted['Pid'].unique()
    ppulls_trial=srted['PPull'].values.reshape(int(srted['PPull'].values.size/len(pids)),len(pids)).T
    ax =sns.heatmap(ppulls_trial,annot=True,xticklabels=[1,2,3,4,5],yticklabels=xlabels)
    ax.set(xlabel='Trial', ylabel='Pids')
    plt.show()


def heatmap_ppull_chal():
    srted=data.sort_values("Pid",0)
    xlabels=srted['Pid'].unique()
    ppulls_trial=srted['PPull'].values.reshape(int(srted['PPull'].values.size/len(pids)),len(pids)).T
    ax =sns.heatmap(ppulls_trial,annot=True,xticklabels=[1,2,3,4,5],yticklabels=xlabels)
    ax.set(xlabel='Trial', ylabel='Pids')
    plt.show()