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
gaze_data =pd.read_csv('D:\\Master\\Thesis\\Results Scripts\\Repo\\gen\\gaze_targets.csv',index_col=False)

misc_data=pd.read_csv('D:\\Master\\Thesis\\Results Scripts\\Repo\\gen\\misc_data.csv')
avatar_rating_data=pd.read_csv('D:\\Master\\Thesis\\Results Scripts\\Repo\\gen\\avatar_rating_data.csv')

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

def misc_plots():
    c.plot(kind='pie', figsize=(8, 8))
   
    
def gaze_plots():
    c=pd.DataFrame(misc_data['VRUse'].value_counts())

    c=pd.DataFrame(gaze_data['GazeTarget'].value_counts())
    c['GazeTarget']=c['GazeTarget']/c['GazeTarget'].sum()
    
    f, ax = plt.subplots()
    c.plot(kind='pie', subplots=True, figsize=(8, 8),labels=None,ax=ax)
    ax.set_ylabel("Gaze % ")
    ax.legend(bbox_to_anchor=(1, 1),labels=c.index)   
    
    sns.set_color_codes("pastel")
    sns.barplot(x="GazeTarget", y="index",data= c, label="Total", color="b")
    
plt.legend(loc='lower right')

def plots():
    df=data[['Condition','Force']].groupby(['Condition']).mean().reset_index()

    #force all data un normalized
    g = sns.catplot(x="Condition", y="Force", kind="swarm", data=data)
  
    sns.lineplot('Condition', 'Force', data=df,marker='x',color="red") 

    g = sns.catplot(x="Condition", y="Force", kind="violin", inner=None, data=data)
    g =sns.swarmplot(x="Condition", y="Force", color="k", size=5, data=data, ax=g.ax)
    g.set_axis_labels("Condition","Force")

    g = sns.lineplot(x="Condition", y="Force",err_style="bars", data=df)
    g.set_yticks(range(0,20))

    #force all data normalized
    g = sns.catplot(x="Condition", y="NForce", kind="swarm", data=data)
    g.set_axis_labels("Condition","NForce")

    ax= sns.catplot(x="Trial", y="NForce", hue="Gender", kind="bar", data=data)
    #height=8.27, aspect=11.7/8.27 height=10
    ax.set_axis_labels("Trial","NForce")


    
def by_cond():
    sns.catplot(x="Condition", y="Challenge", kind="bar", data=data);
    sns.catplot(x="Condition", y="PPull", kind="bar", data=data);

    sns.catplot(x="Condition", y="Challenge",hue='Gender', kind="bar", data=data)
    sns.catplot(x="Condition", y="PPull", hue='Gender',kind="bar", data=data)
    
    sns.catplot(x="Condition", y="Force", kind="bar", data=data);
    sns.catplot(x="Condition", y="Force", hue="Gender", kind="bar", data=data);
    

def by_trial():
    sns.catplot(x="Trial", y="Challenge", kind="bar", data=data)
    sns.catplot(x="Trial", y="PPull", kind="bar", data=data)

    sns.catplot(x="Trial", y="Challenge",hue='Gender', kind="bar", data=data)
    sns.catplot(x="Trial", y="PPull", hue='Gender',kind="bar", data=data)
    
    sns.catplot(x="Trial", y="Force", kind="bar", data=data);
    sns.catplot(x="Trial", y="Force", hue='Gender',kind="bar", data=data);
    


def correlation():
    d = data.sort_values(['Condition','Pid'])
    stats.ttest_ind(d['PPull'],d['Challenge'], equal_var=False)
    stats.ttest_ind(d['PPull'],d['Force'])
    
    
def heatmap_ppull_idx():
    srted=data.sort_values("Trial",0)
    xlabels=srted['Pid'].unique()
    ppulls_trial=srted['PPull'].values.reshape(int(srted['PPull'].values.size/len(pids)),len(pids)).T
    ax =sns.heatmap(ppulls_trial,annot=True,xticklabels=['1st','2nd','3rd','4th','5th'],yticklabels=xlabels,center=5)
    ax.set(xlabel='Trial', ylabel='Pids')
    plt.show()


def heatmap_ppull_chal():
    srted=data.sort_values("Trial",0)
    xlabels=srted['Pid'].unique()
    ppulls_trial=srted['Challenge'].values.reshape(int(srted['Challenge'].values.size/len(pids)),len(pids)).T
    ax =sns.heatmap(ppulls_trial,annot=True,xticklabels=['1 (Weakest)','2','3','4','5 (Strongest)'],yticklabels=xlabels,center=5)
    ax.set(xlabel='Challenge', ylabel='Pids')
    plt.show()
    
    
def stacked_ppull_idx():
    df=pd.DataFrame()
    df.set
    
    data['PPull'].value_counts()
    data[data['Condition']==1]['Challenge'].value_counts() #-get them per condition
    
    data.groupby(['Challenge', 'PPull']).size().unstack().plot(kind='bar', stacked=True)
    df=data[['Challenge','PPull','Trial','Condition']].groupby(['Challenge', 'PPull']).count()
    
    df=data[['Challenge','PPull','Trial','Condition']].groupby(['Condition', 'PPull'])
     
     
idx = np.asarray([i for i in range(5)])
labels = np.asarray([i+1 for i in range(5)])

#for key in df_avatars_f.keys():
#   df_avatars_f[key].plot.barh(stacked = True)

# gets occurence of ratings df_avatars_f[0][df_avatars_f[0].keys()[0]].value_counts()

#category_names = ['1 (Strongly disagree)', '2','3 ','4', ' 5 (Strongly agree)']
#ratings=[1,2,3,4,5]
#levels=[1,2,3,4,5]
#labels = ['Attractive','Strong','Intelligent','Intimidating']
#
#j=0
#for key in df_avatars_f.keys():
#    j=j+1
#    results ={}
#    for key1 in df_avatars_f[key].keys():
#       results[key1]=df_avatars_f[key][key1].value_counts().sort_index().reindex([1,2,3,4,5],fill_value=0).values
#    data = np.array(list(results.values()))
#    data_cum = data.cumsum(axis=1)
#    category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, data.shape[1]))
#    fig, ax = plt.subplots(figsize=(8, 5))
#    ax.invert_yaxis()
#    #ax.xaxis.set_visible(False)
#    ax.set_xlim(0, np.sum(data, axis=1).max())
#    
#    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
#        widths = data[:, i]
#        widths = widths.astype(dtype='float32')
#        starts = data_cum[:, i] - widths
#        ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)
#        #ax.set_ylabel('This avatar looks')
#        ax.set_xlabel('Rating counts')
#        xcenters = starts + widths / 2
#        r, g, b, _ = color
#        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
#        for y, (x, c) in enumerate(zip(xcenters, widths)):
#            if(int(c)!=0):
#                ax.text(x, y, str(int(c)), ha='center', va='center',color=text_color)
#            else:
#                ax.text(x, y, '', ha='center', va='center',color=text_color)
#    fig.subplots_adjust(left=0.2)
#    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),  loc='lower left', fontsize='small')
#    fig.savefig('avatar_f'+str(j)+'.png')
#    plt.close(fig) 
#    