
#rename columns and add condition
renameCols=dict(zip(list(pull_by_condition_normalized.columns.values),list(pull_by_condition.keys().values[1:])))
pull_by_condition_normalized.rename(columns=renameCols,inplace=True)
#add condition colummn back and move it first
pull_by_condition_normalized['Condition']=pull_by_condition['Condition']

newKeys=deque(pull_by_condition_normalized.keys().values.tolist())
newKeys.rotate(1)
pull_by_condition_normalized = pull_by_condition_normalized[list(newKeys)]

index_condition_map=dict(zip(pull_by_condition.index.values.tolist(),pull_by_condition['Condition'].tolist()))
# sns.pairplot(vars=['1','10','5','4','8','9'], data=pull_by_condition)
    

results_full= pd.read_csv("res.csv").drop(columns=['Unnamed: 0'])

result_keys=results_full.keys().values

results_by_index_ppul=pd.DataFrame()
idx_pull_keys = results_by_index_ppul.keys().values

results_by_index_challenge=pd.DataFrame()

results_by_index_ppul['pid']=results_full['pid'].copy()
results_by_index_challenge['pid']=results_full['pid'].copy()

result_chal_pull_per={}
for i in range(0,5):
    results_by_index_ppul[i]=results_full.iloc[:,7+4*i].copy()
    results_by_index_challenge[i]=results_full.iloc[:,8+4*i].copy()
    
    
def scatter_ppul_challenge():
    fig, ax = plt.subplots(figsize=(10, 5))    
    ax.set_ylabel('PPull')
    ax.set_xlabel('Challenge')
    ax.set_xticks(l)    
    for i in range (1,len(idx_pull_keys)):
        ax.scatter( results_by_index_ppul[idx_pull_keys[i]],
                   results_by_index_challenge[idx_pull_keys[i]],
                   color='black')
        #removing top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #adds major gridlines
        ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.show()
    
def bar_index():
    #replace with row 1 below
    plt.style.use('ggplot')
    n = 5
    ppul= results_by_index_ppul.iloc[1,1:]
    challenge = results_by_index_challenge.iloc[1,1:]#
    fig, ax = plt.subplots()
    index = np.arange(n)
    bar_width = 0.35
    opacity = 0.9
    ax.bar(index, ppul, bar_width, alpha=opacity, color='r',label='PPull')
    ax.bar(index+bar_width, challenge, bar_width, alpha=opacity, color='b', label='Chal')
    ax.set_xlabel('Trial #')
    ax.set_ylabel('Rating')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('1st','2nd','3rd','4th','5th'))
    ax.legend()
    plt.show()
    

def scatter_all(jitter=False):
    fig =plt.figure(figsize=(10, 10))    
    gs = gridspec.GridSpec(2, 1) # rows | colums
    gs.update(wspace=0.1, hspace=0.5) # set the spacing between axes. 
    
    ax = fig.add_subplot(gs[0,0]) # Adds subplot 'ax' in grid 'gs' at position [x,y]
    ax1 = fig.add_subplot(gs[1,0]) # Adds subplot 'ax' in grid 'gs' at position [x,y]
    
    for i in range (0,len(pull_keys)):
        if(jitter):
            x=pull_by_condition.sort_values("Condition")['Condition']
            x1=pull_by_condition.index.values
            rand_jitter=np.random.rand(x.shape[0])*width-width/2.
            ax.scatter(x+rand_jitter,pull_by_condition.sort_values("Condition")[pull_keys[i]],color='black',label="Value")
            ax1.scatter(x1+rand_jitter,pull_by_condition.iloc[:,1:][pull_keys[i]],color='black')
        else:
            ax.scatter(pull_by_condition.sort_values("Condition")['Condition'],pull_by_condition.sort_values("Condition")[pull_keys[i]],color='black',label="Value")
            ax1.scatter( pull_by_condition.index.values,pull_by_condition.iloc[:,1:][pull_keys[i]],color='black') 
   
    #plot mean
    ax.plot(pull_by_condition.sort_values("Condition")['Condition'], pull_by_condition.sort_values("Condition").iloc[:,1:].mean(axis=1), '-xk',color='red',label="Mean")       
    ax1.plot( pull_by_condition.index.values, pull_by_condition.iloc[:,1:].mean(axis=1), '-xk',color='red');
    #plot std
    ax.plot(pull_by_condition.sort_values("Condition")['Condition'], pull_by_condition.sort_values("Condition").iloc[:,1:].std(axis=1), '-xk',color='blue',label="Stdev")       
    ax1.plot( pull_by_condition.index.values, pull_by_condition.iloc[:,1:].std(axis=1), '-xk',color='blue');
   
    
    ax.set_ylabel('Force(Kg)')
    ax.set_xlabel('Condition')
    ax.set_xticks(pull_by_condition.index.values+1) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("By Condition");
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
    ax1.set_ylabel('Force(Kg)')
    ax1.set_xlabel('Trial #')
    ax1.set_xticks(pull_by_condition.index.values)    
    ax1.set_xticklabels(pull_by_condition.index.values+1)  
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title("By Index");
    ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    #Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),fontsize=12, loc='upper left', 
              bbox_to_anchor=(0.85,1.25))
    
    plt.show()
  

def scatter_all_normalized(jitter=False):
    fig =plt.figure(figsize=(10, 10))    
    gs = gridspec.GridSpec(2, 1) # rows | colums
    gs.update(wspace=0.1, hspace=0.5) # set the spacing between axes. 
    
    ax = fig.add_subplot(gs[0,0]) # Adds subplot 'ax' in grid 'gs' at position [x,y]
    ax1 = fig.add_subplot(gs[1,0]) # Adds subplot 'ax' in grid 'gs' at position [x,y]
    
    for i in range (0,len(pull_keys)):
        if(jitter):
            x=pull_by_condition_normalized.sort_values("Condition")['Condition']
            x1=pull_by_condition_normalized.index.values
            rand_jitter=np.random.rand(x.shape[0])*width-width/2.
            ax.scatter(x+rand_jitter,pull_by_condition_normalized.sort_values("Condition")[pull_keys[i]],color='black',label="Value")
            ax1.scatter(x1+rand_jitter,pull_by_condition_normalized.iloc[:,1:][pull_keys[i]],color='black')
        else:
            ax.scatter(pull_by_condition_normalized.sort_values("Condition")['Condition'],pull_by_condition_normalized.sort_values("Condition")[pull_keys[i]],color='black',label="Value")
            ax1.scatter( pull_by_condition_normalized.index.values,pull_by_condition_normalized.iloc[:,1:][pull_keys[i]],color='black') 

    #plot mmeans
    ax.plot(pull_by_condition_normalized.sort_values("Condition")['Condition'], pull_by_condition_normalized.sort_values("Condition").iloc[:,1:].mean(axis=1), '-xk',color='red',label="Mean")       
    ax1.plot( pull_by_condition_normalized.index.values, pull_by_condition_normalized.iloc[:,1:].mean(axis=1), '-xk',color='red');
    #plot std
    ax.plot(pull_by_condition_normalized.sort_values("Condition")['Condition'], pull_by_condition_normalized.sort_values("Condition").iloc[:,1:].std(axis=1), '-xk',color='blue',label="Stdev")       
    ax1.plot( pull_by_condition_normalized.index.values, pull_by_condition_normalized.iloc[:,1:].std(axis=1), '-xk',color='blue');

    
    ax.set_ylabel('Force(Kg)')
    ax.set_xlabel('Condition')
    ax.set_xticks(pull_by_condition_normalized.index.values+1) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("By Condition");
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
    ax1.set_ylabel('Force(Kg)')
    ax1.set_xlabel('Trial #')
    ax1.set_xticks(pull_by_condition_normalized.index.values)    
    ax1.set_xticklabels(pull_by_condition_normalized.index.values+1)  
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title("By Index");
    ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    #Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),fontsize=12, loc='upper left', 
              bbox_to_anchor=(0.85,1.25))
    
    plt.show()
  


    