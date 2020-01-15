import numpy as np
import matplotlib
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import matplotlib.patches
import statsmodels.api as sm
import scipy.stats
import pandas as pd
import math
from mpi4py import MPI
import sys
import itertools
plt.ioff()
sys.path.append('../')
from SALib.analyze import delta

design = str(sys.argv[1])

LHsamples = np.loadtxt('../Qgen/' + design + '.txt') 
param_bounds=np.loadtxt('../Qgen/uncertain_params_'+design[10:-5]+'.txt', usecols=(1,2))
all_IDs = np.genfromtxt('../Structures_files/metrics_structures.txt',dtype='str').tolist()
nStructures = len(all_IDs)
params_no = len(LHsamples[0,:])
param_names=[x.split(' ')[0] for x in open('../Qgen/uncertain_params_'+design[10:-5]+'.txt').readlines()]
problem = {
    'num_vars': params_no,
    'names': param_names,
    'bounds': param_bounds.tolist()
}
# Longform parameter names to use in figure legend
parameter_names_long = ['IWR demand mutliplier', 'Reservoir loss', 
                        'TBD demand multiplier', 'M&I demand multiplier', 
                        'Shoshone active', 'Env. flow senior right', 
                        'Evaporation delta', 'Dry state mu', 
                        'Dry state sigma', 'Wet state mu', 
                        'Wet state sigma', 'Dry-to-dry state prob.', 
                        'Wet-to-wet state prob.', 'Earlier snowmelt', 'Interaction']
param_names=['IWRmultiplier','RESloss','TBDmultiplier','M_Imultiplier',
             'ShoshoneDMND','ENVflows','EVAdelta','XBM_mu0','XBM_sigma0',
             'XBM_mu1','XBM_sigma1','XBM_p00','XBM_p11', 'shift']
percentiles = np.arange(0,100)
samples = 1000
realizations = 10
idx = np.arange(2,22,2)

# deal with fact that calling result.summary() in statsmodels.api
# calls scipy.stats.chisqprob, which no longer exists
scipy.stats.chisqprob = lambda chisq, df: scipy.stats.chi2.sf(chisq, df)

def alpha(i, base=0.2):
    l = lambda x: x+base-x*base
    ar = [l(0)]
    for j in range(i):
        ar.append(l(ar[-1]))
    return ar[-1]

def shortage_duration(sequence, threshold):
    cnt_shrt = [sequence[i]>threshold for i in range(len(sequence))] # Returns a list of True values when there's a shortage
    shrt_dur = [ sum( 1 for _ in group ) for key, group in itertools.groupby( cnt_shrt ) if key ] # Counts groups of True values
    return shrt_dur

def fitOLS(dta, predictors):
    # concatenate intercept column of 1s
    dta['Intercept'] = np.ones(np.shape(dta)[0])
    # get columns of predictors
    cols = dta.columns.tolist()[-1:] + predictors
    #fit OLS regression
    ols = sm.OLS(dta['Shortage'], dta[cols])
    result = ols.fit()
    return result

def plotSDC(synthetic, histData, structure_name):
    n = 12
    #Reshape historic data to a [no. years x no. months] matrix
    f_hist = np.reshape(histData, (int(np.size(histData)/n), n))
    #Reshape to annual totals
    f_hist_totals = np.sum(f_hist,1)
    
    #Reshape synthetic data
    #Create matrix of [no. years x no. months x no. samples]
    synthetic_global = np.zeros([int(np.size(histData)/n),n,samples*realizations]) 
    # Loop through every SOW and reshape to [no. years x no. months]
    for j in range(samples*realizations):
        synthetic_global[:,:,j]= np.reshape(synthetic[:,j], (int(np.size(synthetic[:,j])/n), n))
    #Reshape to annual totals
    synthetic_global_totals = np.sum(synthetic_global,1) 

    p=np.arange(100,-10,-10)     
    hist_max_durations = np.zeros(len(percentiles))        
    syth_max_durations = np.zeros([len(percentiles),samples*realizations])
    for i in range(len(percentiles)):
        historic_value = np.percentile(f_hist_totals,percentiles[i])
        historic_durations = shortage_duration(f_hist_totals, historic_value)
        if historic_durations:
            hist_max_durations[i]=np.max(historic_durations)        
        for j in range(samples*realizations):
            synth_durations = shortage_duration(synthetic_global_totals[:,j], historic_value)
            if synth_durations:
                syth_max_durations[i,j] = np.max(synth_durations)
    
    np.save('../'+design+'/Max_Duration_Curves/' + structure_name + '.npy',syth_max_durations)
    np.save('../'+design+'/Max_Duration_Curves/' + structure_name + '_historic.npy',hist_max_durations)
#    hist_max_durations = np.load('../'+design+'/Max_Duration_Curves/' + structure_name + '_historic.npy')    
#    syth_max_durations = np.load('../'+design+'/Max_Duration_Curves/' + structure_name + '.npy')
                   
    ylimit = 105
    fig, (ax1) = plt.subplots(1,1, figsize=(14.5,8))
    # ax1
    handles = []
    labels=[]
    color = '#000292'
    for i in range(len(p)):
        ax1.fill_between(percentiles, np.min(syth_max_durations[:,:],1), np.percentile(syth_max_durations[:,:], p[i], axis=1), color=color, alpha = 0.1)
        ax1.plot(percentiles, np.percentile(syth_max_durations[:,:], p[i], axis=1), linewidth=0.5, color=color, alpha = 0.3)
        handle = matplotlib.patches.Rectangle((0,0),1,1, color=color, alpha=alpha(i, base=0.1))
        handles.append(handle)
        label = "{:.0f} %".format(100-p[i])
        labels.append(label)
    ax1.plot(percentiles,hist_max_durations, c='black', linewidth=2, label='Historical record')
    ax1.set_ylim(0,ylimit)
    ax1.set_xlim(0,100)
    ax1.legend(handles=handles, labels=labels, framealpha=1, fontsize=8, loc='upper left', title='Frequency in experiment',ncol=2)
    ax1.set_xlabel('Shortage magnitude percentile', fontsize=20)
    ax1.set_ylabel('Max shortage duration (years)', fontsize=20)

    fig.suptitle('Shortage max durations for ' + structure_name, fontsize=16)
    plt.subplots_adjust(bottom=0.2)
    fig.savefig('../'+design+'/Max_Duration_Curves/' + structure_name + '.svg')
    fig.savefig('../'+design+'/Max_Duration_Curves/' + structure_name + '.png')
    fig.clf()
    
#    DELTA = pd.DataFrame(np.zeros((params_no, len(percentiles))), columns = [str(p) for p in percentiles])
#    DELTA_conf = pd.DataFrame(np.zeros((params_no, len(percentiles))), columns = [str(p) for p in percentiles])
#    S1 = pd.DataFrame(np.zeros((params_no, len(percentiles))), columns = [str(p) for p in percentiles])
#    S1_conf = pd.DataFrame(np.zeros((params_no, len(percentiles))), columns = [str(p) for p in percentiles])
#    R2_scores = pd.DataFrame(np.zeros((params_no, len(percentiles))), columns = [str(p) for p in percentiles])
#    DELTA.index=DELTA_conf.index=S1.index=S1_conf.index = R2_scores.index = param_names
#    # Delta Method analysis
#    for i in range(len(percentiles)):
#        if syth_max_durations[i,:].any():
#            try:
#                result= delta.analyze(problem, np.repeat(LHsamples, realizations, axis = 0), syth_max_durations[i,:], print_to_console=False, num_resamples=10)
#                DELTA[str(percentiles[i])]= result['delta']
#                DELTA_conf[str(percentiles[i])] = result['delta_conf']
#                S1[str(percentiles[i])]=result['S1']
#                S1_conf[str(percentiles[i])]=result['S1_conf']
#            except:
#                pass
#
#    S1.to_csv('../'+design+'/Max_Duration_Sensitivity_analysis/'+ structure_name + '_S1.csv')
#    S1_conf.to_csv('../'+design+'/Max_Duration_Sensitivity_analysis/'+ structure_name + '_S1_conf.csv')
#    DELTA.to_csv('../'+design+'/Max_Duration_Sensitivity_analysis/'+ structure_name + '_DELTA.csv')
#    DELTA_conf.to_csv('../'+design+'/Max_Duration_Sensitivity_analysis/'+ structure_name + '_DELTA_conf.csv')
#
#    # OLS regression analysis
#    dta = pd.DataFrame(data = np.repeat(LHsamples, realizations, axis = 0), columns=param_names)
#    for i in range(len(percentiles)):
#        dta['Shortage']=syth_max_durations[i,:]
#        for m in range(params_no):
#            predictors = dta.columns.tolist()[m:(m+1)]
#            result = fitOLS(dta, predictors)
#            R2_scores.at[param_names[m],str(percentiles[i])]=result.rsquared
#    R2_scores.to_csv('../'+design+'/Max_Duration_Sensitivity_analysis/'+ structure_name + '_R2.csv')
#  
#    '''
#    Sensitivity analysis plots normalized to 0-100
#    '''
#    
##    DELTA = pd.read_csv('../'+design+'/Magnitude_Sensitivity_analysis/'+ structure_name + '_DELTA.csv')
##    DELTA.set_index(list(DELTA)[0],inplace=True)
#    DELTA = DELTA.clip(lower=0)
#    for p in percentiles:
#        total = np.sum(DELTA[str(p)])
#        if total!=0:
#            for param in param_names:
#                    value = 100*DELTA.at[param,str(p)]/total
#                    DELTA.set_value(param,str(p),value)
#    delta_values_to_plot = DELTA.values.tolist()
#    
##    S1 = pd.read_csv('../'+design+'/Magnitude_Sensitivity_analysis/'+ structure_name + '_S1.csv')
##    S1.set_index(list(S1)[0],inplace=True)
#    S1 = S1.clip(lower=0)
#    for p in percentiles:
#        total = np.sum(S1[str(p)])
#        if total!=0 and total<1:
#            diff = 1-total
#            S1.at['Interaction',str(p)]=diff
#        else:
#            S1.at['Interaction',str(p)]=0               
#    for column in S1:
#        S1[column] = S1[column]*100
#    S1_values_to_plot = S1.values.tolist()
#
##    R2_scores = pd.read_csv('../'+design+'/Magnitude_Sensitivity_analysis/'+ structure_name + '_R2.csv')
##    R2_scores.set_index(list(R2_values)[0],inplace=True)
#    R2_scores = R2_scores.clip(lower=0)
#    for p in percentiles:
#        total = np.sum(R2_scores[str(p)])
#        if total!=0 and total<1:
#            diff = 1-total
#            R2_scores.at['Interaction',str(p)]=diff
#        else:
#            R2_scores.at['Interaction',str(p)]=0  
#    for column in R2_scores:
#        R2_scores[column] = R2_scores[column]*100
#    R2_values_to_plot = R2_scores.values.tolist()
#    
#    color_list = ["#F18670", "#E24D3F", "#CF233E", "#681E33", "#676572", "#F3BE22", "#59DEBA", "#14015C", "#DAF8A3", "#0B7A0A", "#F8FFA2", "#578DC0", "#4E4AD8", "#32B3F7","#F77632"]  
#                  
#    values_to_plot = [delta_values_to_plot, S1_values_to_plot, R2_values_to_plot]
#    titles = ["Delta","S1","R2"]
#    yaxistitles = ["Change explained","Variance explained","Variance explained"]
#    for k in range(len(titles)):
#        fig, (ax1) = plt.subplots(1,1, figsize=(14.5,8))
#        ax1.stackplot(np.arange(0,100), values_to_plot[k], colors = color_list)
#        ax1.set_title(titles[k])
#        ax1.set_ylim(0,150)
#        ax1.set_xlim(0,100)
#        handles, labels = ax1.get_legend_handles_labels()
#        ax1.set_ylabel(yaxistitles[k], fontsize=12)
#        ax1.set_xlabel('Shortage magnitude percentile', fontsize=12)
#        plt.legend(handles, labels = parameter_names_long, fontsize=10, loc='lower center',ncol = 5)
#        fig.suptitle('Sensitivity of max duration of each shortage percentile for '+ structure_name, fontsize=16)
#        fig.savefig('../'+design+'/Max_Duration_Curves_SA/' + structure_name + '_'+titles[k]+'_norm.svg')
#        fig.savefig('../'+design+'/Max_Duration_Curves_SA/' + structure_name + '_'+titles[k]+'_norm.png')


# Begin parallel simulation
comm = MPI.COMM_WORLD

# Get the number of processors and the rank of processors
rank = comm.rank
nprocs = comm.size

# Determine the chunk which each processor will neeed to do
count = int(math.floor(nStructures/nprocs))
remainder = nStructures % nprocs

# Use the processor rank to determine the chunk of work each processor will do
if rank < remainder:
	start = rank*(count+1)
	stop = start + count + 1
else:
	start = remainder*(count+1) + (rank-remainder)*count
	stop = start + count
    
for i in range(start, stop):
    histData = np.loadtxt('../'+design+'/Infofiles/' +  all_IDs[i] + '/' + all_IDs[i] + '_info_0.txt')[:,2]*1233.4818/1000000
    synthetic = np.zeros([len(histData), samples*realizations])
    for j in range(samples):
        data= np.loadtxt('../'+design+'/Infofiles/' +  all_IDs[i] + '/' + all_IDs[i] + '_info_' + str(j+1) + '.txt') 
        try:
            synthetic[:,j*realizations:j*realizations+realizations]=data[:,idx]*1233.4818/1000000
        except IndexError:
            print(all_IDs[i] + '_info_' + str(j+1))
    plotSDC(synthetic, histData, all_IDs[i])

    
