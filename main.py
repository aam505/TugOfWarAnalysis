import matplotlib.pyplot as plt
import pandas as pd     
import numpy as np
import glob
import seaborn as sns
from sklearn import preprocessing
import matplotlib.gridspec as gridspec
from collections import OrderedDict 
from collections import deque
from scipy import stats
from numpy import median
from matplotlib import rc

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
gaze_data =pd.read_csv('D:\\Master\\Thesis\\Results Scripts\\Repo\\gen\\other\\gaze_targets.csv',index_col=False)

misc_data=pd.read_csv('D:\\Master\\Thesis\\Results Scripts\\Repo\\gen\\other\\misc_data.csv')
avatar_rating_data=pd.read_csv('D:\\Master\\Thesis\\Results Scripts\\Repo\\gen\\other\\avatar_rating_data.csv')

pids = list(pull_data.keys()) 
#pids=[int(x) for x in pids]

#data.to_csv('D:\\Master\\Thesis\\Results Scripts\\Repo\\data.csv')

#normalize between 0,1  
min_max_scaler = preprocessing.MinMaxScaler()

#Make pull by condition and pull by force tables for all participants
for i in range (0,len(pids)):
     for key in pull_data[pids[i]].keys():
         if(key!='Gender' and key!='Pid'):
             pull_data[pids[i]][key] = pull_data[pids[i]][key].astype('int64')
     pull_data[pids[i]]['ForceNormalized'] = min_max_scaler.fit_transform(pd.DataFrame(pull_data[pids[i]]['Force'].values))
     #pull_data[pids[i]]['MaxDifference'] = pd.DataFrame(pull_data[pids[i]]['Force'].values)
     pull_data[pids[i]]['MaxForceDif']=pull_data[pids[i]]['Force'].values.max()-pull_data[pids[i]]['Force'].values.min()


#aggregate
data=pull_data[pids[0]].copy()
for i in range (1,len(pids)):
    data=data.append(pull_data[pids[i]], ignore_index = True)

first = data.groupby('Pid')['Force'].first()
def norm_force(x):
    return x['Force'] / first.loc[x['Pid']]
data['NForce'] = data[['Pid', 'Force']].apply(norm_force, axis=1)


#data.to_csv('D:\\Master\\Thesis\\Results Scripts\\Repo\\data.csv')
data_stable_maxdif=data[data['MaxForceDif']<12] #3 users

def misc_plots():

    ratings=[1,2,3,4,5]
    #misc_data.groupby('VRUse').size().plot(kind='pie', figsize=(8, 8),autopct='%1.1f%%',startangle=90,shadow=False, labels=list(misc_data.groupby('VRUse').size().index))
    keys =  misc_data.keys()
    presence_data=misc_data[[keys[5],keys[6],keys[7],keys[8],keys[9]]]
    presence_data=presence_data.rename(columns={keys[5]:'Q1',keys[6]:'Q2',keys[7]:'Q3',keys[8]:'Q4',keys[9]:'Q5'})
    
    sns.set(font_scale=1.5)    
    ax= sns.catplot( kind="bar", data=presence_data,color='orange',ci='sd')
    ax.set(xlabel='Question', ylabel='Mean Rating and Std')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    
    sns.set(font_scale=1.5)    
    ax= sns.catplot( kind="bar", data=presence_data,hue='gender',ci='sd')
    ax.set(xlabel='Question', ylabel='Mean Rating and Std')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    
    
    presence_mean = presence_data.mean().reset_index()
    presence_std= presence_data.std().reset_index()
    presence_median = presence_data.median().reset_index()
    
    presence_mean=presence_mean.rename(index={0: 'Q1',1:'Q2',2:'Q3',3:'Q4',4:'Q5'},columns={'index':'q'}).reset_index()
    presence_std=presence_std.rename(index={0: 'Q1',1:'Q2',2:'Q3',3:'Q4',4:'Q5'},columns={'index':'q'}).reset_index()
    presence_median=presence_median.rename(index={0: 'Q1',1:'Q2',2:'Q3',3:'Q4',4:'Q5'},columns={'index':'q'}).reset_index()

    presence_data.to_numpy().sum()

    sns.set(font_scale=1.5)    
    ax = presence_median.plot.bar(x='index',legend=False,yerr=presence_std[0].values)
    ax.set_xlabel('Question')
    ax.set_ylabel("Median Rating and Std")
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    sns.set(font_scale=1.5)    
    ax = presence_mean.plot.bar(x='index',legend=False,yerr=presence_std[0].values)
    ax.set_xlabel('Question')
    ax.set_ylabel("Mean Rating and Std")
    plt.xticks(rotation=0)
    plt.tight_layout()

    ownership_data=misc_data[[keys[10],keys[11],keys[12],keys[13],keys[14],keys[15]]]

    ownership_mean = ownership_data.mean().reset_index()
    ownership_std= ownership_data.std().reset_index()
    ownership_median = ownership_data.median().reset_index()
    
    ownership_mean=ownership_mean.rename(index={0: 'Q1',1:'Q2',2:'Q3',3:'Q4',4:'Q5',5:'Q6'},columns={'index':'q'}).reset_index()
    ownership_std=ownership_std.rename(index={0: 'Q1',1:'Q2',2:'Q3',3:'Q4',4:'Q5',5:'Q6'},columns={'index':'q'}).reset_index()
    ownership_median=ownership_median.rename(index={0: 'Q1',1:'Q2',2:'Q3',3:'Q4',4:'Q5',5:'Q6'},columns={'index':'q'}).reset_index()
 
    sns.set(font_scale=1.5)    
    ax = ownership_median.plot.bar(x='index',legend=False,yerr=ownership_std[0].values)
    ax.set_xlabel('Question')
    ax.set_ylabel("Median Rating and Std")
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    sns.set(font_scale=1.5)    
    ax = ownership_mean.plot.bar(x='index',legend=False,yerr=ownership_std[0].values)
    ax.set_xlabel('Question')
    ax.set_ylabel("Mean Rating and Std")
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    ownership_data.to_numpy().mean()
    
    copresence_data=misc_data[[keys[16],keys[17],keys[18]]]

    copresence_mean = copresence_data.mean().reset_index()
    copresence_std= copresence_data.std().reset_index()
    copresence_median = copresence_data.median().reset_index()
    
    copresence_mean=copresence_mean.rename(index={0: 'Q1',1:'Q2',2:'Q3'},columns={'index':'q'}).reset_index()
    copresence_std=copresence_std.rename(index={0: 'Q1',1:'Q2',2:'Q3'},columns={'index':'q'}).reset_index()
    copresence_median=copresence_median.rename(index={0: 'Q1',1:'Q2',2:'Q3'},columns={'index':'q'}).reset_index()
    
    sns.set(font_scale=1.5)    
    ax = copresence_median.plot.bar(x='index',legend=False,yerr=copresence_std[0].values)
    ax.set_xlabel('Question')
    ax.set_ylabel("Median Rating and Std")
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    sns.set(font_scale=1.5)    
    ax = copresence_mean.plot.bar(x='index',legend=False,yerr=copresence_std[0].values)
    ax.set_xlabel('Question')
    ax.set_ylabel("Mean Rating and Std")
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    copresence_data.to_numpy().mean()
    
    
    pd_all=pd.DataFrame()
    pd_all['Co-presence']=np.array([copresence_data.to_numpy().mean()])
    pd_all['Presence']=np.array([ presence_data.to_numpy().mean()])
    pd_all['Ownership']=np.array([ownership_data.to_numpy().mean()])
    
    yerr= np.array( [copresence_data.to_numpy().std(),presence_data.to_numpy().std(),ownership_data.to_numpy().std()])    
    
    sns.set(font_scale=1.5)    
    ax =  pd_all.T.reset_index().plot.bar(x='index',legend=False,yerr=yerr,align='center', alpha=0.5, ecolor='black', capsize=10,rot=0)
    plt.tight_layout()
    ax.set_xlabel('Category')
    ax.set_ylabel("Total mean rating and std")
    
    g = sns.catplot(x="Trial", y="NForce", kind="bar", data=presence_data)
    g.set_ylabel("Gaze % ")



def misc_plots_gender_female():

    ratings=[1,2,3,4,5]
    #misc_data.groupby('VRUse').size().plot(kind='pie', figsize=(8, 8),autopct='%1.1f%%',startangle=90,shadow=False, labels=list(misc_data.groupby('VRUse').size().index))
    keys =  misc_data.keys()
    presence_data=misc_data[misc_data['Gender']=='Female'][[keys[5],keys[6],keys[7],keys[8],keys[9]]]
    presence_data=presence_data.rename(columns={keys[5]:'Q1',keys[6]:'Q2',keys[7]:'Q3',keys[8]:'Q4',keys[9]:'Q5'})
    
    
    presence_mean = presence_data.mean().reset_index()
    presence_std= presence_data.std().reset_index()
    presence_median = presence_data.median().reset_index()
    
    presence_mean=presence_mean.rename(index={0: 'Q1',1:'Q2',2:'Q3',3:'Q4',4:'Q5'},columns={'index':'q'}).reset_index()
    presence_std=presence_std.rename(index={0: 'Q1',1:'Q2',2:'Q3',3:'Q4',4:'Q5'},columns={'index':'q'}).reset_index()
    presence_median=presence_median.rename(index={0: 'Q1',1:'Q2',2:'Q3',3:'Q4',4:'Q5'},columns={'index':'q'}).reset_index()

    presence_data.to_numpy().sum()

    sns.set(font_scale=1.5)    
    ax = presence_median.plot.bar(x='index',legend=False,yerr=presence_std[0].values,color='LightBlue')
    ax.set_xlabel('Question')
    ax.set_ylabel("Median Rating and Std")
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    sns.set(font_scale=1.5)    
    ax = presence_mean.plot.bar(x='index',legend=False,yerr=presence_std[0].values,color='LightBlue')
    ax.set_xlabel('Question')
    ax.set_ylabel("Mean Rating and Std")
    plt.xticks(rotation=0)
    plt.tight_layout()

    ownership_data=misc_data[misc_data['Gender']=='Female'][[keys[10],keys[11],keys[12],keys[13],keys[14],keys[15]]]

    ownership_mean = ownership_data.mean().reset_index()
    ownership_std= ownership_data.std().reset_index()
    ownership_median = ownership_data.median().reset_index()
    
    ownership_mean=ownership_mean.rename(index={0: 'Q1',1:'Q2',2:'Q3',3:'Q4',4:'Q5',5:'Q6'},columns={'index':'q'}).reset_index()
    ownership_std=ownership_std.rename(index={0: 'Q1',1:'Q2',2:'Q3',3:'Q4',4:'Q5',5:'Q6'},columns={'index':'q'}).reset_index()
    ownership_median=ownership_median.rename(index={0: 'Q1',1:'Q2',2:'Q3',3:'Q4',4:'Q5',5:'Q6'},columns={'index':'q'}).reset_index()
 
    sns.set(font_scale=1.5)    
    ax = ownership_mean.plot.bar(x='index',legend=False,yerr=ownership_std[0].values,color='LightBlue')
    ax.set_xlabel('Question')
    ax.set_ylabel("Mean Rating and Std")
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    ownership_data.to_numpy().mean()
    
    copresence_data=misc_data[misc_data['Gender']=='Female'][[keys[16],keys[17],keys[18]]]

    copresence_mean = copresence_data.mean().reset_index()
    copresence_std= copresence_data.std().reset_index()
    copresence_median = copresence_data.median().reset_index()
    
    copresence_mean=copresence_mean.rename(index={0: 'Q1',1:'Q2',2:'Q3'},columns={'index':'q'}).reset_index()
    copresence_std=copresence_std.rename(index={0: 'Q1',1:'Q2',2:'Q3'},columns={'index':'q'}).reset_index()
    copresence_median=copresence_median.rename(index={0: 'Q1',1:'Q2',2:'Q3'},columns={'index':'q'}).reset_index()
    
    sns.set(font_scale=1.5)    
    ax = copresence_mean.plot.bar(x='index',legend=False,yerr=copresence_std[0].values,color='LightBlue')
    ax.set_xlabel('Question')
    ax.set_ylabel("Mean Rating and Std")
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    copresence_data.to_numpy().mean()
    
    
    pd_all=pd.DataFrame()
    pd_all['Co-presence']=np.array([copresence_data.to_numpy().mean()])
    pd_all['Presence']=np.array([ presence_data.to_numpy().mean()])
    pd_all['Ownership']=np.array([ownership_data.to_numpy().mean()])
    
    yerr= np.array( [copresence_data.to_numpy().std(),presence_data.to_numpy().std(),ownership_data.to_numpy().std()])    
    
    sns.set(font_scale=1.5)    
    ax =  pd_all.T.reset_index().plot.bar(x='index',legend=False,yerr=yerr,align='center', alpha=0.5, ecolor='black', capsize=10,rot=0,color='LightBlue')
    plt.tight_layout()
    ax.set_xlabel('Category')
    ax.set_ylabel("Total mean rating and std")


  
def misc_plots_gender_male():

    ratings=[1,2,3,4,5]
    #misc_data.groupby('VRUse').size().plot(kind='pie', figsize=(8, 8),autopct='%1.1f%%',startangle=90,shadow=False, labels=list(misc_data.groupby('VRUse').size().index))
    keys =  misc_data.keys()
    presence_data=misc_data[misc_data['Gender']=='Male'][[keys[5],keys[6],keys[7],keys[8],keys[9]]]
    presence_data=presence_data.rename(columns={keys[5]:'Q1',keys[6]:'Q2',keys[7]:'Q3',keys[8]:'Q4',keys[9]:'Q5'})
    
    
    presence_mean = presence_data.mean().reset_index()
    presence_std= presence_data.std().reset_index()
    presence_median = presence_data.median().reset_index()
    
    presence_mean=presence_mean.rename(index={0: 'Q1',1:'Q2',2:'Q3',3:'Q4',4:'Q5'},columns={'index':'q'}).reset_index()
    presence_std=presence_std.rename(index={0: 'Q1',1:'Q2',2:'Q3',3:'Q4',4:'Q5'},columns={'index':'q'}).reset_index()
    presence_median=presence_median.rename(index={0: 'Q1',1:'Q2',2:'Q3',3:'Q4',4:'Q5'},columns={'index':'q'}).reset_index()

    presence_data.to_numpy().sum()
    
    sns.set(font_scale=1.5)    
    ax = presence_mean.plot.bar(x='index',legend=False,yerr=presence_std[0].values,color='orange')
    ax.set_xlabel('Question')
    ax.set_ylabel("Mean Rating and Std")
    plt.xticks(rotation=0)
    plt.tight_layout()

    ownership_data=misc_data[misc_data['Gender']=='Male'][[keys[10],keys[11],keys[12],keys[13],keys[14],keys[15]]]

    ownership_mean = ownership_data.mean().reset_index()
    ownership_std= ownership_data.std().reset_index()
    ownership_median = ownership_data.median().reset_index()
    
    ownership_mean=ownership_mean.rename(index={0: 'Q1',1:'Q2',2:'Q3',3:'Q4',4:'Q5',5:'Q6'},columns={'index':'q'}).reset_index()
    ownership_std=ownership_std.rename(index={0: 'Q1',1:'Q2',2:'Q3',3:'Q4',4:'Q5',5:'Q6'},columns={'index':'q'}).reset_index()
    ownership_median=ownership_median.rename(index={0: 'Q1',1:'Q2',2:'Q3',3:'Q4',4:'Q5',5:'Q6'},columns={'index':'q'}).reset_index()
 

    
    sns.set(font_scale=1.5)    
    ax = ownership_mean.plot.bar(x='index',legend=False,yerr=ownership_std[0].values,color='orange')
    ax.set_xlabel('Question')
    ax.set_ylabel("Mean Rating and Std")
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    ownership_data.to_numpy().mean()
    
    copresence_data=misc_data[misc_data['Gender']=='Male'][[keys[16],keys[17],keys[18]]]

    copresence_mean = copresence_data.mean().reset_index()
    copresence_std= copresence_data.std().reset_index()
    copresence_median = copresence_data.median().reset_index()
    
    copresence_mean=copresence_mean.rename(index={0: 'Q1',1:'Q2',2:'Q3'},columns={'index':'q'}).reset_index()
    copresence_std=copresence_std.rename(index={0: 'Q1',1:'Q2',2:'Q3'},columns={'index':'q'}).reset_index()
    copresence_median=copresence_median.rename(index={0: 'Q1',1:'Q2',2:'Q3'},columns={'index':'q'}).reset_index()
    
    sns.set(font_scale=1.5)    
    ax = copresence_mean.plot.bar(x='index',legend=False,yerr=copresence_std[0].values,color='orange')
    ax.set_xlabel('Question')
    ax.set_ylabel("Mean Rating and Std")
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    copresence_data.to_numpy().mean()
    
    
    pd_all=pd.DataFrame()
    pd_all['Co-presence']=np.array([copresence_data.to_numpy().mean()])
    pd_all['Presence']=np.array([ presence_data.to_numpy().mean()])
    pd_all['Ownership']=np.array([ownership_data.to_numpy().mean()])
    
    yerr= np.array( [copresence_data.to_numpy().std(),presence_data.to_numpy().std(),ownership_data.to_numpy().std()])    
    
    sns.set(font_scale=1.5)    
    ax =  pd_all.T.reset_index().plot.bar(x='index',legend=False,yerr=yerr,align='center', alpha=0.5, ecolor='black', capsize=10,rot=0,color='orange')
    plt.tight_layout()
    ax.set_xlabel('Category')
    ax.set_ylabel("Total mean rating and std")

  
def gaze_plots():
    c=pd.DataFrame(gaze_data['GazeTarget'].value_counts())
    c['GazeTarget']=c['GazeTarget']/(c['GazeTarget'].sum())
    
    sns.set(font_scale=2)  
    f, ax = plt.subplots()
    c.plot(kind='pie', subplots=True, figsize=(8, 8),labels=None,ax=ax, 
           autopct='%1.0f%%', pctdistance=1.2, labeldistance=2, rot=0)
    ax.set_ylabel("Gaze % ")
    ax.legend(bbox_to_anchor=(1.2, 1),labels=c.index)   
    
    sns.set_color_codes("pastel")
    sns.barplot(x="GazeTarget", y="index",data= c, label="Total", color="b")
    
sns.set(style="whitegrid")
def plots():
    #force all data un normalized
    g = sns.catplot(x="Condition", y="Force", kind="swarm", data=data)
    
    g = sns.catplot(x="Condition", y="Force", kind="violin", inner=None, data=data)
    g =sns.swarmplot(x="Condition", y="Force", color="k", size=5, data=data, ax=g.ax)
    g.set_axis_labels("Condition","Force")

    #force all data normalized
    g = sns.catplot(x="Condition", y="NForce", kind="swarm", data=data)
    g = sns.catplot(x="Condition", y="ForceNormalized", kind="swarm", data=data)
   
    
    ax= sns.catplot(x="Trial", y="Force", kind="bar", data=data)
    ax= sns.catplot(x="Trial", y="ForceNormalized",kind="bar", data=data)
    ax= sns.catplot(x="Trial", y="NForce", kind="bar", data=data)
    
    ax= sns.catplot(x="Condition", y="Force", kind="bar", data=data)
    ax= sns.catplot(x="Condition", y="ForceNormalized",kind="bar", data=data)
    ax= sns.catplot(x="Condition", y="NForce", kind="bar", data=data)
    
    
    ax= sns.catplot(x="Trial", y="Force", hue="Gender", kind="bar", data=data)
    ax= sns.catplot(x="Trial", y="ForceNormalized", hue="Gender", kind="bar", data=data)
    ax= sns.catplot(x="Trial", y="NForce", hue="Gender", kind="bar", data=data)
    
    ax= sns.catplot(x="Condition", y="Force", hue="Gender", kind="bar", data=data)
    ax= sns.catplot(x="Condition", y="ForceNormalized", hue="Gender", kind="bar", data=data)
    ax= sns.catplot(x="Condition", y="NForce", hue="Gender", kind="bar", data=data)
    
    ax= sns.catplot(x="Condition", y="RopeRealism", kind="bar", data=data)
    sns.catplot(x="Condition", y="RopeRealism", kind="bar", data=data,hue='Gender')

    
    ax= sns.catplot(x="Trial", y="RopeRealism", kind="bar", data=data)
    sns.catplot(x="Trial", y="RopeRealism", kind="bar", data=data,hue='Gender')
        
    ax= sns.catplot(x="Condition", y="RopeOwnership", kind="bar",data=data)
    ax= sns.catplot(x="Condition", y="RopeOwnership", kind="bar", data=data,hue='Gender',legend='Full')
   
    
    ax= sns.catplot(x="Trial", y="RopeOwnership", kind="bar",data=data)
    ax.legend()
    ax= sns.catplot(x="Trial", y="RopeOwnership", kind="bar", data=data,hue='Gender')
    
    
    sns.lineplot(x='Pid',y='MaxForceDif', data=data_stable_maxdif)
    
    sns.catplot(x="Pid", y="MaxForceDif", kind="bar",hue="Gender",data=data_stable_maxdif)
    
    sns.catplot(x="Pid", y="MaxForceDif", kind="bar",color="purple",data=data_stable_maxdif)


    ax = sns.scatterplot(x="Pid", y="MaxForceDif", hue="Gender", data=data_stable_maxdif)
    ax = sns.scatterplot(x="Pid", y="MaxForceDif",   data=data_stable_maxdif)
        
    
    sns.regplot(x=data_stable_maxdif["Pid"], y=data_stable_maxdif["MaxForceDif"])

def by_cond():

    sns.catplot(x="Condition", y="Challenge", kind="bar", data=data,estimator=np.median,hue='Gender')
    sns.catplot(x="Condition", y="Challenge", kind="bar", data=data,hue='Gender')
    
    sns.catplot(x="Condition", y="PPull", kind="bar", data=data,estimator=np.median,hue='Gender')
    sns.catplot(x="Condition", y="PPull", kind="bar", data=data,hue='Gender')
    
    
    sns.catplot(x="Trial", y="Challenge", kind="bar", data=data,estimator=np.median,hue='Gender')
    sns.catplot(x="Trial", y="Challenge", kind="bar", data=data,hue='Gender')
    
    sns.catplot(x="Trial", y="PPull", kind="bar", data=data,estimator=np.median,hue='Gender')
    sns.catplot(x="Trial", y="PPull", kind="bar", data=data,hue='Gender')
    
    sns.catplot(x="Condition", y="Challenge", kind="bar", data=data,hue='Gender')
    sns.catplot(x="Condition", y="PPull", kind="bar", data=data)
    
        
    sns.catplot(x="Trial", y="Challenge", kind="bar", data=data)
    sns.catplot(x="Trial", y="PPull", kind="bar", data=data)
    
    sns.catplot(x="Condition", y="PPull", kind="bar", data=data,estimator=np.median)
    
    
    sns.catplot(x="Condition", y="Challenge", kind="bar", data=data,estimator=np.mean)

    sns.countplot(x="Condition",  hue='Challenge',data=data,palette="ch:2.5,-.2,dark=.3")
    sns.countplot(x="Condition",  hue='PPull',data=data,palette="ch:2.5,-.2,dark=.3")

    sns.catplot(x="Condition", y="PPull", kind="bar", data=data);

    sns.catplot(x="Condition", y="Challenge",hue='Gender', kind="bar", data=data)
    sns.catplot(x="Condition", y="PPull", hue='Gender',kind="bar", data=data)
    
    sns.catplot(x="Trial", y="Challenge", kind="bar", data=data)
    sns.catplot(x="Trial", y="PPull", kind="bar", data=data)

    sns.catplot(x="Trial", y="Challenge",hue='Gender', kind="bar", data=data)
    sns.catplot(x="Trial", y="PPull", hue='Gender',kind="bar", data=data)
    
    ax= sns.catplot(x="Condition", y="ForceNormalized", kind="bar", data=data[data['Trial']==1],hue='Gender')
    ax= sns.catplot(x="Condition", y="ForceNormalized", kind="bar", data=data[data['Trial']==1],hue='Gender')
    
    ax= sns.catplot(x="Condition", y="PPull", kind="bar", data=data[data['Trial']==1],hue='Gender')
    ax= sns.catplot(x="Condition", y="PPull", kind="bar", data=data[data['Trial']==1])

    ax= sns.catplot(x="Condition", y="NForce", kind="bar", data=data[data['Trial']==1])
    
    ax= sns.catplot(x="Condition", y="ForceNormalized", kind="bar", data=data[data['Trial']==1],hue='Gender')


    ax= sns.catplot(x="Condition", y="ForceNormalized", kind="bar", data=data[data['Trial']==1],hue='Gender')
    
    
    sns.catplot(x="Condition", y="ForceNormalized", kind="bar", data=data[data['Trial'].isin([1,2,3])],hue='Gender')
    
    sns.catplot(x="Condition", y="Trial", kind="bar", data=data[data['Trial'].isin([1,2,3])])
    sns.catplot(x="Condition", y="PPull", kind="bar", data=data[data['Trial'].isin([1,2,3])],hue='Gender')
    
    
    sns.countplot(x="Trial", hue="Condition",data=data)

    sns.set(font_scale=3)
    g=sns.countplot(palette="RdBu_r",x="Trial", hue="Condition",data=data)
    g.legend( bbox_to_anchor=(0.94, 0.6))


    
    sns.set(font_scale=1.5)
    sns.countplot(x="Condition", hue="Trial",data=data[data['Gender']=='female'])
    sns.countplot(x="Trial", hue="Condition",data=data)
    sns.countplot(x="Condition", y="Trial", kind="bar", data=data,hue='Gender')
     
    ax= sns.catplot(x="Condition", y="Force", kind="bar", data=data[data['Trial']==4])

def stacked_challenge():
 
    labels = ['1 (Weak)','2','3 (Average)','4','5 (Strong)']
    ratings=[1,2,3,4,5]
    levels=[1,2,3,4,5]
    category_names = ['1 (Not at all)','2','3','4','5 (Very challenging)']

    a=data.groupby('Condition')['Challenge'].value_counts()
    
    cond1=[0,0,0,0,0]
    cond2=[0,0,0,0,0]
    cond3=[0,0,0,0,0]
    cond4=[0,0,0,0,0]
    cond5=[0,0,0,0,0]
    
    for key in a.keys():
        if (key[0]==1):
            cond1[key[1]-1] = a[key]
        if (key[0]==2):
            cond2[key[1]-1] = a[key]
        if (key[0]==3):
            cond3[key[1]-1] = a[key]
        if (key[0]==4):
            cond4[key[1]-1] = a[key]              
        if (key[0]==5):
            cond5[key[1]-1] = a[key]    
            
    data1 = np.array(list([cond1,cond2,cond3,cond4,cond5]))
    data_cum = data1.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, data1.shape[1]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.invert_yaxis()
    #ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data1, axis=1).max())
    
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data1[:, i]
        widths = widths.astype(dtype='float32')
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)
        #ax.set_ylabel('This avatar looks')
        xcenters = starts + widths / 2
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if(int(c)!=0):
                ax.text(x, y, str(int(c)), ha='center', va='center',color=text_color)
            else:
                ax.text(x, y, '', ha='center', va='center',color=text_color)
    fig.subplots_adjust(left=0.2)
    ax.set_ylabel("Condition")
    ax.set_xlabel('Challenging rating counts')
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),  loc='lower left', fontsize='small')

def stacked_ppull():
    labels = ['1 (Weak)','2','3 (Average)','4','5 (Strong)']
    ratings=[1,2,3,4,5]
    levels=[1,2,3,4,5]
    category_names = ['1 (Not at all)','2','3','4','5 (Very much)']

    a=data.groupby('Condition')['PPull'].value_counts()
    
    cond1=[0,0,0,0,0]
    cond2=[0,0,0,0,0]
    cond3=[0,0,0,0,0]
    cond4=[0,0,0,0,0]
    cond5=[0,0,0,0,0]
    
    for key in a.keys():
        if (key[0]==1):
            cond1[key[1]-1] = a[key]
        if (key[0]==2):
            cond2[key[1]-1] = a[key]
        if (key[0]==3):
            cond3[key[1]-1] = a[key]
        if (key[0]==4):
            cond4[key[1]-1] = a[key]              
        if (key[0]==5):
            cond5[key[1]-1] = a[key]    
            
    data1 = np.array(list([cond1,cond2,cond3,cond4,cond5]))
    data_cum = data1.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, data1.shape[1]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.invert_yaxis()
    #ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data1, axis=1).max())
    
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data1[:, i]
        widths = widths.astype(dtype='float32')
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)
        #ax.set_ylabel('This avatar looks')
        xcenters = starts + widths / 2
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if(int(c)!=0):
                ax.text(x, y, str(int(c)), ha='center', va='center',color=text_color)
            else:
                ax.text(x, y, '', ha='center', va='center',color=text_color)
    fig.subplots_adjust(left=0.2)
    ax.set_ylabel("Condition")
    ax.set_xlabel('PPull rating counts')
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),  loc='lower left', fontsize='small')

def correlation():
    d = data.sort_values(['Condition','Pid'])
    stats.ttest_ind(d['PPull'],d['Challenge'], equal_var=False)
    stats.ttest_ind(d['PPull'],d['Force'])
    
    
def heatmap_ppull_idx():

    srted=data.sort_values(["Condition",'Pid'],0)
    xlabels=srted['Pid'].unique()
    ppulls_trial=srted['PPull'].values.reshape(int(srted['PPull'].values.size/len(pids)),len(pids)).T
    ax =sns.heatmap(ppulls_trial,annot=True,xticklabels=['1 (Weakest)','2','3','4','5 (Strongest)'],yticklabels=xlabels,center=5)
    ax.set(xlabel='Condition', ylabel='Pids')
    plt.show()


def heatmap_chal():
    srted=data.sort_values(["Condition",'Pid'],0)
    xlabels=srted['Pid'].unique().T
    ppulls_trial=srted['Challenge'].values.reshape(int(srted['Challenge'].values.size/len(pids)),len(pids)).T
    ax =sns.heatmap(ppulls_trial,annot=True,xticklabels=['1 (Weakest)','2','3','4','5 (Strongest)'],yticklabels=xlabels,center=5)
    ax.set(xlabel='Condition', ylabel='Pids')
    plt.show()
    
  