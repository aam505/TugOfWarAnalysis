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


condition_mapping={0:5,1:3,2:1,3:4, 4:2}
df = pd.read_csv("appearance.csv") 

df_f= df[df['gender'] == 'Female']
#df_f = df_f.drop('Pid', 1)
df_f = df_f.drop('Timestamp', 1)
df_f = df_f.drop('gender', 1)
df_f = df_f.drop('age', 1)

df_f=df_f.dropna(axis=1)

df_avatars_f={}
for i in range(0,len(df_f.columns),4):
    df_avatars_f[i]=df_f.iloc[:, i:(i+4)]   
    
#df_avatars= df_f.iloc[:, 0:4]    
ticks=[i for i in range (0,18)]

i=1

idx = np.asarray([i for i in range(5)])
labels = np.asarray([i+1 for i in range(5)])

#for key in df_avatars_f.keys():
#   df_avatars_f[key].plot.barh(stacked = True)

# gets occurence of ratings df_avatars_f[0][df_avatars_f[0].keys()[0]].value_counts()

category_names = ['1 (Strongly disagree)', '2','3 ','4', ' 5 (Strongly agree)']
ratings=[1,2,3,4,5]
levels=[1,2,3,4,5]
labels = ['Attractive','Strong','Intelligent','Intimidating']

j=0
for key in df_avatars_f.keys():
    j=j+1
    results ={}
    for key1 in df_avatars_f[key].keys():
       results[key1]=df_avatars_f[key][key1].value_counts().sort_index().reindex([1,2,3,4,5],fill_value=0).values
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, data.shape[1]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.invert_yaxis()
    #ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        widths = widths.astype(dtype='float32')
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)
        #ax.set_ylabel('This avatar looks')
        ax.set_xlabel('Rating counts')
        xcenters = starts + widths / 2
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if(int(c)!=0):
                ax.text(x, y, str(int(c)), ha='center', va='center',color=text_color)
            else:
                ax.text(x, y, '', ha='center', va='center',color=text_color)
    fig.subplots_adjust(left=0.2)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),  loc='lower left', fontsize='small')
 
    #fig.savefig('avatar_f_experiment'+str(condition_mapping[j-1])+'.png')
    #fig.savefig('avatar_f'+str(j)+'.png')
    break
    #plt.close(fig) 
    
#for key in df_avatars_f.keys():
#    fig = plt.figure()
#    
#    ax1 = fig.add_subplot(221)
#    df_avatars_f[key][df_avatars_f[key].columns[0]].value_counts().plot.bar(ax=ax1)
#    ax1.set_title(df_avatars_f[0].columns[0])
#    ax1.set_ylabel('Count')
#    ax1.set_xlabel('Rating')
#    ax1.set_xticks(idx)
#    ax1.set_xticklabels(labels)
#    
#    ax2 = fig.add_subplot(222)
#    df_avatars_f[key][df_avatars_f[key].columns[1]].value_counts().plot.bar(ax=ax2)
#    ax2.set_title(df_avatars_f[0].columns[1])
#    ax2.set_ylabel('Count')
#    ax2.set_xlabel('Rating')
#    ax2.set_xticks(idx)
#    ax2.set_xticklabels(labels)
#    
#    ax3 =fig.add_subplot(223)
#    df_avatars_f[key][df_avatars_f[key].columns[2]].value_counts().plot.bar(ax=ax3)
#    ax3.set_title(df_avatars_f[0].columns[2])
#    ax3.set_ylabel('Count')
#    ax3.set_xlabel('Rating')
#    ax3.set_xticks(idx)
#    ax3.set_xticklabels(labels)
#    
#    ax4 = fig.add_subplot(224)
#    df_avatars_f[key][df_avatars_f[key].columns[3]].value_counts().plot.bar(ax=ax4)
#    ax4.set_title(df_avatars_f[0].columns[3])
#    ax4.set_ylabel('Count')
#    ax4.set_xlabel('Rating')
#    ax4.set_xticks(idx)
#    ax4.set_xticklabels(labels)
#    
#    fig.subplots_adjust(hspace=0.9,wspace=0.9)
#    fig.savefig('avatar_f'+str(i)+'.png')
#    i=i+1
#    plt.close(fig) 
#
#
df_m= df.loc[df['gender'] == 'Male']
#df_m = df_m.drop('Pid', 1)

df_m = df_m.drop('Timestamp', 1)
df_m = df_m.drop('gender', 1)
df_m = df_m.drop('age', 1)

df_m=df_m.dropna(axis=1, how='all')

df_avatars_m={}
for i in range(0,len(df_m.columns),4):
    df_avatars_m[i]=df_m.iloc[:, i:(i+4)]  
    
#i=1
#for key in df_avatars_m.keys():
#    fig = plt.figure()
#    ax1 = fig.add_subplot(221)
#    df_avatars_m[key][df_avatars_m[key].columns[0]].value_counts().plot.bar(ax=ax1)
#    ax1.set_title(df_avatars_f[0].columns[0])
#    ax1.set_ylabel('Count')
#    ax1.set_xlabel('Rating')
#    ax1.set_xticks(idx)
#    ax1.set_xticklabels(labels)
#    
#    ax2 = fig.add_subplot(222)
#    df_avatars_m[key][df_avatars_m[key].columns[1]].value_counts().plot.bar(ax=ax2)
#    ax2.set_title(df_avatars_f[0].columns[1])
#    ax2.set_ylabel('Count')
#    ax2.set_xlabel('Rating')
#    ax2.set_xticks(idx)
#    ax2.set_xticklabels(labels)
#    
#    ax3 =fig.add_subplot(223)
#    df_avatars_m[key][df_avatars_m[key].columns[2]].value_counts().plot.bar(ax=ax3)
#    ax3.set_title(df_avatars_f[0].columns[2])
#    ax3.set_ylabel('Count')
#    ax3.set_xlabel('Rating')
#    ax3.set_xticks(idx)
#    ax3.set_xticklabels(labels)
#    
#    ax4 = fig.add_subplot(224)
#    df_avatars_m[key][df_avatars_m[key].columns[3]].value_counts().plot.bar(ax=ax4)
#    ax4.set_title(df_avatars_f[0].columns[3])
#    ax4.set_ylabel('Count')
#    ax4.set_xlabel('Rating')
#    ax4.set_xticks(idx)
#    ax4.set_xticklabels(labels)
#    
#    fig.subplots_adjust(hspace=0.9,wspace=0.9)
#    fig.savefig('avatar_m'+str(i)+'.png')
#    i=i+1
#    plt.close(fig) 
#

    
#i=0
#mean_f=[]
#mean_m=[]
#
#for key in df_avatars_f.keys():
#    mean_f.append({})  
#    mean_f[i]["attractive"]=df_avatars_f[key][df_avatars_f[key].columns[0]].mean()
#    mean_f[i]["strong"]=df_avatars_f[key][df_avatars_f[key].columns[1]].mean()
#    mean_f[i]["intelligent"]=df_avatars_f[key][df_avatars_f[key].columns[2]].mean()
#    mean_f[i]["intimidating"]=df_avatars_f[key][df_avatars_f[key].columns[3]].mean()
#    
#    mean_m.append({}) 
#    mean_m[i]["attractive"]=df_avatars_m[key][df_avatars_m[key].columns[0]].mean()
#    mean_m[i]["strong"]=df_avatars_m[key][df_avatars_m[key].columns[1]].mean()
#    mean_m[i]["intelligent"]=df_avatars_m[key][df_avatars_m[key].columns[2]].mean()
#    mean_m[i]["intimidating"]=df_avatars_m[key][df_avatars_m[key].columns[3]].mean()
#    
#    i=i+1
#    
#    
#df_f = pd.DataFrame(mean_f)
#df_m = pd.DataFrame(mean_m)
#
#df_f.to_csv (r'mean_rating_females.csv', index = None, header=True) 
#df_m.to_csv (r'mean_rating_males.csv', index = None, header=True) 
    
j=0
for key in df_avatars_m.keys():
    j=j+1
    results ={}
    for key1 in df_avatars_m[key].keys():
       results[key1]=df_avatars_m[key][key1].value_counts().sort_index().reindex([1,2,3,4,5],fill_value=0).values
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, data.shape[1]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.invert_yaxis()
    #ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        widths = widths.astype(dtype='float32')
        #widths[widths==0]=0.1
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)
        #ax.set_ylabel('This avatar looks')
        ax.set_xlabel('Rating counts')
        xcenters = starts + widths / 2
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if(int(c)!=0):
                ax.text(x, y, str(int(c)), ha='center', va='center',color=text_color)
            else:
                ax.text(x, y, '', ha='center', va='center',color=text_color)
    fig.subplots_adjust(left=0.2)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),  loc='lower left', fontsize='small')
    break
    #fig.savefig('avatar_m'+str(j)+'.png')
    #fig.savefig('avatar_m_experiment'+str(condition_mapping[j-1])+'.png')
    #plt.close(fig) 
    


pids=list(df_avatars_f.keys())


data_f=df_avatars_f[pids[0]].copy()
data_m=df_avatars_m[pids[0]].copy()

data_m.insert(0,'ID',[1] * data_m.index.size)
data_f.insert(0,'ID',[1] * data_f.index.size)

data_m.columns = data_m.columns.str.split('.').str[0]
data_f.columns = data_f.columns.str.split('.').str[0]


#data_m.insert(0,'Condition',[condition_mapping[0]] * data_m.index.size)
#data_f.insert(0,'Condition',[condition_mapping[0]] * data_f.index.size)

for i in range (1,len(pids)):
    
    df1=df_avatars_f[pids[i]].copy()
    df1.insert(0,'ID',[i+1] * df1.index.size)
    df1.columns = df1.columns.str.split('.').str[0]

    #for experiment
    #df1.insert(0,'Condition',[condition_mapping[i]] * df1.index.size)
    
    df2=df_avatars_m[pids[i]].copy()
    df2.columns = df2.columns.str.split('.').str[0]
    df2.insert(0,'ID',[i+1] * df2.index.size)

    data_f=data_f.append(df1, ignore_index = True)
    data_m=data_m.append(df2, ignore_index = True)
    
data_f['Weighted']=data_f['This avatar looks strong']*0.5 + data_f['This avatar looks intimidating']*0.5 
    
data_m['Weighted']=data_m['This avatar looks strong']*0.5 + data_m['This avatar looks intimidating']*0.5 
    
def catplots():
    #experiment
    #sns.catplot(x="Condition", y="Weighted", kind="bar", data=data_f)
    #sns.catplot(x="Condition", y="Weighted", kind="bar", data=data_m)
    
    #survey
    chosen_f=data_f[data_f['ID']==10]
    chosen_f=chosen_f.append(data_f[data_f['ID']==16])
    chosen_f=chosen_f.append(data_f[data_f['ID']==18])
    chosen_f=chosen_f.append(data_f[data_f['ID']==5])
    chosen_f=chosen_f.append(data_f[data_f['ID']==2])
    
    sns.catplot(x="ID", y="Weighted", kind="bar", data=chosen_f,order=[10,16,18,5,2])
    
    chosen_m=data_m[data_m['ID']==13]
    chosen_m=chosen_m.append(data_m[data_m['ID']==12])
    chosen_m=chosen_m.append(data_m[data_m['ID']==7])
    chosen_m=chosen_m.append(data_m[data_m['ID']==3])
    chosen_m=chosen_m.append(data_m[data_m['ID']==2])
    
    
    sns.catplot(x="ID", y="Weighted", kind="bar", data=chosen_m,order=[13,12,7,3,2])
    #sns.catplot(x="ID", y="Weighted", kind="bar", data=data_m)
    #sns.catplot(x="ID", y="Weighted", kind="bar", data=data_f)
    
    #survey chosen
    
    
    