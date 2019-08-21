import matplotlib.pyplot as plt
import pandas as pd     
import numpy as np
import os
import glob
from collections import deque

def swapPositions(list, pos1, pos2): 
      
    list[pos1], list[pos2] = list[pos2], list[pos1] 
    return list

cond_mapping={'Average':'c2','Strong':'c1','Weak':'c3',
                'Average UMA_F2':2,
                'Strong UMA_F4':5,
                'Strong UMA_F1':4,
                'Average UMA_F3':3,
                'Weak UMA_F4':1,
                
                'Strong UMA_M5':5,
                'Average UMA_M1':4,
                'Strong UMA_M6':3,
                'Weak UMA_M7':1,
                'Average UMA_M7':2}
basepath="D:\\Master\\Thesis\\Results Scripts\\logs\\**"

def filebrowser(word=""):
    """Returns a list with all files with the word/extension in it"""
    file = []
    
    for f in glob.glob(basepath):
        if word in f:
            file.append({f[38:-4]:f})
    return file


def load_csvs(path):
    p_dict={}
    p_dict_error={}
    """Returns a list with all files with the word/extension in it"""
    word="csv"
    for f in glob.glob(path):
        if word in f:
            df=pd.read_csv(f,index_col=False )[['Condition']]['Condition'].drop_duplicates().dropna()#keep top 5
            if(df.shape[0]<5):
                print("Id:"+str(f[38:-4])+" conditions badly formed: " + str(df.shape[0]))
                p_dict_error[f[38:-4]] =  df
            else:
                p_dict[f[38:-4]] =  df[:5].reset_index().drop(columns=['index'])
                if(df.shape[0]>5):
                     print("Warning at id:"+str(f[38:-4])+" check validity ")
                     p_dict_error[f[38:-4]] =  df
        
    return p_dict,p_dict_error

trial_cond_order,_=load_csvs(basepath)


for key in trial_cond_order.keys():
    trial_cond_order[key]['Pid']=[key]*5
    trial_cond_order[key]['Trial']=trial_cond_order[key].index+1
    if('F' in trial_cond_order[key]['Condition'][0]):
        trial_cond_order[key]['Gender']=['female']*5
    else:
        trial_cond_order[key]['Gender']=['male']*5

    trial_cond_order[key]['Condition']=trial_cond_order[key]['Condition'].apply(lambda x: cond_mapping[x])
    #arrange order of columns
    newKeys=swapPositions(trial_cond_order[key].keys().values.tolist(),0,1)
    trial_cond_order[key] = trial_cond_order[key][list(newKeys)]
    trial_cond_order[key]['Force']=['']

#
for key in trial_cond_order.keys():
    trial_cond_order[key].to_csv('\\Generated\\'+str(key)+'.csv')

