import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt 
plt.switch_backend('agg')
from mpi4py import MPI
import math
import os
plt.ioff()
import sys

# =============================================================================
# Experiment set up
# =============================================================================
design = str(sys.argv[1])

LHsamples = np.loadtxt('../Qgen/' + design + '.txt')  
# convert streamflow multipliers/deltas in LHsamples to HMM parameter values
def convertMultToParams(multipliers):
    params = np.zeros(np.shape(multipliers))
    params[:,0] = multipliers[:,0]*15.258112 # historical dry state mean
    params[:,1] = multipliers[:,1]*0.259061 # historical dry state std dev
    params[:,2] = multipliers[:,2]*15.661007 # historical wet state mean
    params[:,3] = multipliers[:,3]*0.252174 # historical wet state std dev
    params[:,4] = multipliers[:,4] + 0.679107 # historical dry-dry transition prob
    params[:,5] = multipliers[:,5] + 0.649169 # historical wet-wet transition prob
    
    return params

# convert HMM parameter values to streamflow multipliers/deltas in LHsamples 
def convertParamsToMult(params):
    multipliers = np.zeros(np.shape(params))
    multipliers[:,0] = params[:,0]/15.258112 # historical dry state mean
    multipliers[:,1] = params[:,1]/0.259061 # historical dry state std dev
    multipliers[:,2] = params[:,2]/15.661007 # historical wet state mean
    multipliers[:,3] = params[:,3]/0.252174 # historical wet state std dev
    multipliers[:,4] = params[:,4] - 0.679107 # historical dry-dry transition prob
    multipliers[:,5] = params[:,5] - 0.649169 # historical wet-wet transition prob
    
    return multipliers

# find SOWs where mu0 > mu1 and swap their wet and dry state parameters
HMMparams = convertMultToParams(LHsamples[:,[7,8,9,10,11,12]])
for i in range(np.shape(HMMparams)[0]):
    if HMMparams[i,0] > HMMparams[i,2]: # dry state mean above wet state mean
        # swap dry and wet state parameters to correct labels
        mu0 = HMMparams[i,2]
        std0 = HMMparams[i,3]
        mu1 = HMMparams[i,0]
        std1 = HMMparams[i,1]
        p00 = HMMparams[i,5]
        p11 = HMMparams[i,4]
        newParams = np.array([[mu0, std0, mu1, std1, p00, p11]])
        LHsamples[i,[7,8,9,10,11,12]] = convertParamsToMult(newParams)

realizations = 10
param_bounds=np.loadtxt('../Qgen/uncertain_params_'+design[10:-5]+'.txt', usecols=(1,2))
SOW_values = np.array([1,1,1,1,0,0,1,1,1,1,1,0,0,0]) #Default parameter values for base SOW
params_no = len(LHsamples[0,:])
param_names=['IWRmultiplier','RESloss','TBDmultiplier','M_Imultiplier',
             'Shoshone','ENVflows','EVAdelta','XBM_mu0','XBM_sigma0',
             'XBM_mu1','XBM_sigma1','XBM_p00','XBM_p11', 'shift']

all_IDs = np.genfromtxt('../Structures_files/metrics_structures.txt',dtype='str').tolist()
nStructures = len(all_IDs)

# deal with fact that calling result.summary() in statsmodels.api
# calls scipy.stats.chisqprob, which no longer exists
scipy.stats.chisqprob = lambda chisq, df: scipy.stats.chi2.sf(chisq, df)

n=12 # Number of months in a year for reshaping data
nMonths = n * 105 #Record is 105 years long

idx_shortage = np.arange(2,22,2)
idx_demand = np.arange(1,21,2)

# Set arrays for shortage frequencies and magnitudes
frequencies = np.arange(10, 110, 10)
magnitudes = np.arange(10, 110, 10)
streamdemand = np.array([1950, 1750, 1630, 1450, 1240, 1150, 950, 810, 650, 500])*60.3707

def roundup(x):
    return int(math.ceil(x / 10.0)) * 10

def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def plotfailureheatmap(ID):
    if ID=='7202003':
        streamflow = np.zeros([nMonths,len(LHsamples[:,0])*realizations])
        
        for j in range(len(LHsamples[:,0])):
            data= np.loadtxt('../'+design+'/Infofiles/' +\
                             ID + '/' + ID + '_streamflow_' + str(j+1) + '.txt')[:,1:]
            streamflow[:,j*realizations:j*realizations+realizations]=data
        
        streamflowhistoric = np.loadtxt('../'+design+'/Infofiles/' +\
                                        ID + '/' + ID + '_streamflow_0.txt')[:,1]
        
        historic_percents = [100-scipy.stats.percentileofscore(streamflowhistoric, dem, kind='strict') for dem in streamdemand]

        percentSOWs = np.zeros([len(frequencies), len(streamdemand)])
        allSOWs = np.zeros([len(frequencies), len(streamdemand), len(LHsamples[:,0])*realizations])
        
        for j in range(len(frequencies)):
            for h in range(len(streamdemand)):
                success = np.zeros(len(LHsamples[:,0])*realizations)
                for k in range(len(success)):
                    if 100 - scipy.stats.percentileofscore(streamflow[:,k], streamdemand[h], kind='strict')>(100-frequencies[j]):
                        success[k]=100
                allSOWs[j,h,:]=success
                percentSOWs[j,h]=np.mean(success)
        gridcells = []
        for x in historic_percents:
            try:
                gridcells.append(list(frequencies)[::-1].index(roundup(x)))
            except:
                gridcells.append(0)
        
    else:                     
        data= np.loadtxt('../'+design+'/Infofiles/' +  ID + '/' + ID + '_info_0.txt')
        historic_demands = data[:,1]
        historic_shortages = data[:,2]
        #reshape into water years
        historic_shortages_f= np.reshape(historic_shortages, (int(np.size(historic_shortages)/n), n))
        historic_demands_f= np.reshape(historic_demands, (int(np.size(historic_demands)/n), n))
        
        historic_demands_f_WY = np.sum(historic_demands_f,axis=1)
        historic_shortages_f_WY = np.sum(historic_shortages_f,axis=1)
        historic_ratio = (historic_shortages_f_WY*100)/historic_demands_f_WY
        historic_percents = [100-scipy.stats.percentileofscore(historic_ratio, mag, kind='strict') for mag in magnitudes]
        
        percentSOWs = np.zeros([len(frequencies), len(magnitudes)])
        allSOWs = np.zeros([len(frequencies), len(magnitudes), len(LHsamples[:,0])*realizations])
        
        shortages = np.zeros([nMonths,len(LHsamples[:,0])*realizations])
        demands = np.zeros([nMonths,len(LHsamples[:,0])*realizations])
        for j in range(len(LHsamples[:,0])):
            data= np.loadtxt('../'+design+'/Infofiles/' +  ID + '/' + ID + '_info_' + str(j+1) + '.txt')
            try:
                demands[:,j*realizations:j*realizations+realizations]=data[:,idx_demand]
                shortages[:,j*realizations:j*realizations+realizations]=data[:,idx_shortage]
            except:
                print('problem with ' + ID + '_info_' + str(j+1))
                
        #Reshape into water years
        #Create matrix of [no. years x no. months x no. experiments]
        f_shortages = np.zeros([int(nMonths/n),n,len(LHsamples[:,0])*realizations])
        f_demands = np.zeros([int(nMonths/n),n,len(LHsamples[:,0])*realizations]) 
        for i in range(len(LHsamples[:,0])*realizations):
            f_shortages[:,:,i]= np.reshape(shortages[:,i], (int(np.size(shortages[:,i])/n), n))
            f_demands[:,:,i]= np.reshape(demands[:,i], (int(np.size(demands[:,i])/n), n))
        
        # Shortage per water year
        f_demands_WY = np.sum(f_demands,axis=1)
        f_shortages_WY = np.sum(f_shortages,axis=1)
        for j in range(len(frequencies)):
            for h in range(len(magnitudes)):
                success = np.zeros(len(LHsamples[:,0])*realizations)
                for k in range(len(success)):
                    ratio = (f_shortages_WY[:,k]*100)/f_demands_WY[:,k]
                    if scipy.stats.percentileofscore(ratio, magnitudes[h], kind='strict')>(100-frequencies[j]):
                        success[k]=100
                allSOWs[j,h,:]=success
                percentSOWs[j,h]=np.mean(success)
                
        gridcells = []
        for x in historic_percents:
            try:
                gridcells.append(list(frequencies).index(roundup(x)))
            except:
                gridcells.append(0)
    
    fig, ax = plt.subplots()
    im = ax.imshow(percentSOWs, norm = mpl.colors.Normalize(vmin=0.0,vmax=100.0), cmap='RdBu', interpolation='nearest')
#    for j in range(len(frequencies)):
#        for h in range(len(magnitudes)):
#            ax.text(h, j, int(percentSOWs[j,h]),
#                           ha="center", va="center", color="w")
    
    if ID=='7202003':
        ax.set_xticks(np.arange(len(streamdemand)))
        ax.set_xticklabels([str(int(x/60.3707)) for x in list(streamdemand)])
        ax.set_xlabel("Streamflow to meet (cfs)")
        ax.set_ylabel("Percent of time flow is met")
        ax.set_yticks(np.arange(len(frequencies)))
        ax.set_yticklabels([str(x) for x in list(frequencies)[::-1]])        
    else:
        ax.set_xticks(np.arange(len(magnitudes)))
        ax.set_xticklabels([str(x) for x in list(magnitudes)])
        ax.set_xlabel("Percent of demand that is short")
        ax.set_ylabel("Percent of time shortage is experienced")
        ax.set_yticks(np.arange(len(frequencies)))
        ax.set_yticklabels([str(x) for x in list(frequencies)])
        
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(percentSOWs.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(percentSOWs.shape[0]+1)-.5, minor=True)
    #ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    cbar = ax.figure.colorbar(im, ax=ax, cmap='RdBu')
    cbar.ax.set_ylabel("Percentage of SOWs where criterion was met", rotation=-90, va="bottom")
    
    for i in range(len(historic_percents)):
        if historic_percents[i]!=0:
            highlight_cell(i,gridcells[i], color="orange", linewidth=4)
    
    fig.tight_layout()
    fig.savefig('../'+design+'/Factor_mapping/Heatmaps/'+ID+'.svg')
    plt.close()
    
    np.save('../'+design+'/Factor_mapping/'+ ID + '_heatmap.npy',allSOWs)
    
    return(allSOWs, historic_percents)       

def fitLogit(dta, predictors):
    # concatenate intercept column of 1s
    dta['Intercept'] = np.ones(np.shape(dta)[0]) 
    # get columns of predictors
    cols = dta.columns.tolist()[-1:] + predictors 
    #fit logistic regression
    logit = sm.Logit(dta['Success'], dta[cols], disp=False)
    result = logit.fit() 
    return result  

def plotContourMap(ax, result, dta, contour_cmap, dot_cmap, levels, xgrid, ygrid, \
    xvar, yvar, base):
 
    # find probability of success for x=xgrid, y=ygrid
    X, Y = np.meshgrid(xgrid, ygrid)
    x = X.flatten()
    y = Y.flatten()
    grid = np.column_stack([np.ones(len(x)),x,y])
 
    z = result.predict(grid)
    Z = np.reshape(z, np.shape(X))

    contourset = ax.contourf(X, Y, Z, levels, cmap=contour_cmap)
    ax.scatter(dta[xvar].values, dta[yvar].values, c=dta['Success'].values, edgecolor='none', cmap=dot_cmap)
    ax.set_xlim(0.99*np.min(X),1.01*np.max(X))
    ax.set_ylim(0.99*np.min(Y),1.01*np.max(Y))
    ax.set_xlabel(xvar,fontsize=10)
    ax.set_ylabel(yvar,fontsize=10)
    ax.tick_params(axis='both',labelsize=6)
    return contourset

#def shortage_duration(sequence, value):
#    cnt_shrt = [sequence[i]>value for i in range(len(sequence))] # Returns a list of True values when there's a shortage about the value
#    shrt_dur = [ sum( 1 for _ in group ) for key, group in itertools.groupby( cnt_shrt ) if key ] # Counts groups of True values
#    return shrt_dur

def factor_mapping(ID):
    allSOWsperformance, historic_percents = plotfailureheatmap(ID)
    allSOWsperformance = allSOWsperformance/100
    historic_percents = [roundup(x) for x in historic_percents]
    dta = pd.DataFrame(data = np.repeat(LHsamples, realizations, axis = 0), columns=param_names)
    if not os.path.exists('../'+design+'/Factor_mapping/LR_contours/'+ ID):
        os.makedirs('../'+design+'/Factor_mapping/LR_contours/'+ ID)
    if ID!='7202003':
        all_pseudo_r_scores = pd.DataFrame()
        for j in range(len(frequencies)):
            for h in range(len(magnitudes)):
                pseudo_r_scores = np.zeros(params_no)
                # Logistic regression analysis
                dta['Success']=allSOWsperformance[j,h,:]
                for m in range(params_no):
                    predictors = dta.columns.tolist()[m:(m+1)]
                    try:
                        result = fitLogit(dta, predictors)
                        pseudo_r_scores[m]=result.prsquared
                    except: 
                        pseudo_r_scores[m]=pseudo_r_scores[m]
                all_pseudo_r_scores[str(frequencies[j])+'yrs_'+str(magnitudes[h])+'prc']=pseudo_r_scores
        all_pseudo_r_scores.to_csv('../'+design+'/Factor_mapping/'+ ID + '_pseudo_r_scores.csv', sep=",")
        for h in range(len(magnitudes)):
            if historic_percents[h]!=0:
                dta['Success']=allSOWsperformance[list(frequencies).index(historic_percents[h]),h,:]
                pseudo_r_scores = all_pseudo_r_scores[str(int(historic_percents[h]))+'yrs_'+str(magnitudes[h])+'prc'].values
                if pseudo_r_scores.any():
                    fig, axes = plt.subplots(1,1)
                    top_predictors = np.argsort(pseudo_r_scores)[::-1][:2] #Sort scores and pick top 2 predictors
                    # define color map for dots representing SOWs in which the policy
                    # succeeds (light blue) and fails (dark red)
                    dot_cmap = mpl.colors.ListedColormap(np.array([[227,26,28],[166,206,227]])/255.0)
                    # define color map for probability contours
                    contour_cmap = mpl.cm.get_cmap('RdBu')
                    # define probability contours
                    contour_levels = np.arange(0.0, 1.05,0.1)
                    # define base values of the predictors
                    base = SOW_values[top_predictors]
                    ranges = param_bounds[top_predictors]
                    # define grid of x (1st predictor), and y (2nd predictor) dimensions
                    # to plot contour map over
                    xgrid = np.arange(param_bounds[top_predictors[0]][0], 
                                      param_bounds[top_predictors[0]][1], np.around((ranges[0][1]-ranges[0][0])/100,decimals=4))
                    ygrid = np.arange(param_bounds[top_predictors[1]][0], 
                                      param_bounds[top_predictors[1]][1], np.around((ranges[1][1]-ranges[1][0])/100,decimals=4))
                    all_predictors = [ dta.columns.tolist()[i] for i in top_predictors]
                    result = fitLogit(dta, [all_predictors[i] for i in [0,1]])
                    contourset = plotContourMap(axes, result, dta, contour_cmap, 
                                                dot_cmap, contour_levels, xgrid, 
                                                ygrid, all_predictors[0], all_predictors[1], base)
                    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                    cbar = fig.colorbar(contourset, cax=cbar_ax)
                    cbar_ax.set_ylabel('Probability',fontsize=12)
                    yticklabels = cbar.ax.get_yticklabels()
                    cbar.ax.set_yticklabels(yticklabels,fontsize=10)
                    fig.set_size_inches([14.5,8])
                    fig.suptitle('Probability of not having a '+ str(magnitudes[h]) +\
                                 ' shortage ' +  str(frequencies[j]) + '% of the time for '+ ID)
                    fig.savefig('../'+design+'/Factor_mapping/LR_contours/'+\
                                ID+'/'+str(int(historic_percents[h]))+'yrsw'+str(magnitudes[h])+'pcshortm.svg')
                    plt.close()
                        
    elif ID=='7202003':
        all_pseudo_r_scores = pd.DataFrame()
        for j in range(len(frequencies)):
            for h in range(len(streamdemand)):
                pseudo_r_scores = np.zeros(params_no)
                # Logistic regression analysis
                dta = pd.DataFrame(data = np.repeat(LHsamples, realizations, axis = 0), columns=param_names)
                dta['Success']=allSOWsperformance[j,h,:]
                for m in range(params_no):
                    predictors = dta.columns.tolist()[m:(m+1)]
                    try:
                        result = fitLogit(dta, predictors)
                        pseudo_r_scores[m]=result.prsquared
                    except: 
                        pseudo_r_scores[m]=pseudo_r_scores[m]
                all_pseudo_r_scores[str(frequencies[::-1][j])+'yrs_'+str(int(streamdemand[h]/60.3707))+'prc']=pseudo_r_scores
                if pseudo_r_scores.any():
                    fig, axes = plt.subplots(1,1)
                    top_predictors = np.argsort(pseudo_r_scores)[::-1][:2] #Sort scores and pick top 2 predictors
                    # define color map for dots representing SOWs in which the policy
                    # succeeds (light blue) and fails (dark red)
                    dot_cmap = mpl.colors.ListedColormap(np.array([[227,26,28],[166,206,227]])/255.0)
                    # define color map for probability contours
                    contour_cmap = mpl.cm.get_cmap('RdBu')
                    # define probability contours
                    contour_levels = np.arange(0.0, 1.05,0.1)
                    # define base values of the predictors
                    base = SOW_values[top_predictors]
                    ranges = param_bounds[top_predictors]
                    # define grid of x (1st predictor), and y (2nd predictor) dimensions
                    # to plot contour map over
                    xgrid = np.arange(param_bounds[top_predictors[0]][0], param_bounds[top_predictors[0]][1], np.around((ranges[0][1]-ranges[0][0])/100,decimals=4))
                    ygrid = np.arange(param_bounds[top_predictors[1]][0], param_bounds[top_predictors[1]][1], np.around((ranges[1][1]-ranges[1][0])/100,decimals=4))
                    all_predictors = [ dta.columns.tolist()[i] for i in top_predictors]
                    result = fitLogit(dta, [all_predictors[i] for i in [0,1]])
                    contourset = plotContourMap(axes, result, dta, contour_cmap, 
                                                dot_cmap, contour_levels, xgrid, 
                                                ygrid, all_predictors[0], all_predictors[1], base)
                    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                    cbar = fig.colorbar(contourset, cax=cbar_ax)
                    cbar_ax.set_ylabel('Probability',fontsize=12)
                    yticklabels = cbar.ax.get_yticklabels()
                    cbar.ax.set_yticklabels(yticklabels,fontsize=10)
                    fig.set_size_inches([14.5,8])
                    fig.suptitle('Probability of not having a '+ str(int(streamdemand[h]/60.3707)) +\
                                 ' cf/s flow ' +  str(frequencies[::-1][j]) + '% of the time for '+ ID)
                    fig.savefig('../'+design+'/Factor_mapping/LR_contours/'+\
                                ID+'/'+str(frequencies[::-1][j])+'yrsw'+str(int(streamdemand[h]/60.3707))+'pcshortm.png')
                    fig.savefig('../'+design+'/Factor_mapping/LR_contours/'+\
                                ID+'/'+str(frequencies[::-1][j])+'yrsw'+str(int(streamdemand[h]/60.3707))+'pcshortm.svg')                    
                    plt.close()
        all_pseudo_r_scores.to_csv('../'+design+'/Factor_mapping/'+ ID + '_pseudo_r_scores.csv', sep=",")                        
        

    
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
    
# Run simulation
for i in range(start, stop):
    factor_mapping(all_IDs[i])