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
basepath="E:\\Google Drive\\Thesis\\Experiment Log\\**"

def load_csvs(path):
    p_dict={}
    p_dict_error={}
    """Returns a list with all files with the word/extension in it"""
    word="csv"
    offset=38
    for f in glob.glob(path):
        if word in f:
            df=pd.read_csv(f,index_col=False )[['Condition']]['Condition'].drop_duplicates().dropna()#keep top 5
            if(df.shape[0]<5):
                print("Id:"+str(f[offset:-4])+" conditions badly formed: " + str(df.shape[0]))
                p_dict_error[f[offset:-4]] =  df
            else:
                p_dict[f[offset:-4]]=  df[:5].reset_index().drop(columns=['index'])
                if(df.shape[0]>5):
                     print("Warning at id:"+str(f[offset:-4])+" check validity ")
                     p_dict_error[f[offset:-4]] =  df
        
    return p_dict,p_dict_error

trial_cond_order,war=load_csvs(basepath)

survey_data= pd.read_csv("D:\\Master\\Thesis\\Results Scripts\\Repo\\res.csv")

percep_data=pd.DataFrame()
percep_data['Pid']=""
percep_data['Trial'] = ""
percep_data['PPull'] = ""
percep_data['Challenge'] = ""
percep_data['RopeRealism'] = ""
percep_data['RopeOwnership'] = ""

for i in range(0,5):
    df=pd.DataFrame()
    df['Pid']=survey_data['Pid']
    df['Trial']=[i+1]*len(survey_data.index.values.tolist())
    df['RopeRealism']=survey_data.iloc[:,5+4*i].copy()
    df['RopeOwnership']=survey_data.iloc[:,6+4*i].copy()
    df['PPull']=survey_data.iloc[:,7+4*i].copy()
    df['Challenge']=survey_data.iloc[:,8+4*i].copy()
    percep_data=percep_data.append(df.copy())

percep_data = percep_data[['Trial', 'Pid','RopeRealism', 'RopeOwnership', 'PPull','Challenge']]

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
    trial_cond_order[key]['Force']=''
    trial_cond_order[key]['Trial'] = trial_cond_order[key]['Trial'].astype('int64')
    trial_cond_order[key]['Pid'] = trial_cond_order[key]['Pid'].astype('int64')
    
    trial_cond_order[key]=pd.merge(trial_cond_order[key],percep_data[percep_data['Pid']==int(key)],on=['Pid','Trial'])

def log():
    for key in trial_cond_order.keys():
        trial_cond_order[key].to_csv('D:\\Master\\Thesis\\Results Scripts\\Repo\\gen\\'+str(key)+'.csv')


