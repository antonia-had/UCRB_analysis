import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt 
import sys
plt.switch_backend('agg')
plt.ioff()

#design = str(sys.argv[1])
design = 'LHsamples_original_1000'

LHsamples = np.loadtxt('../../Qgen/' + design + '.txt')

def convertLogMultToRealMult(logMultipliers):
    # mu0, sigma0, mu1, sigma1, p00, p11
    # first four columns are multipliers in log-space; conver to multipliers in real-space
    realMultipliers = np.zeros(np.shape(logMultipliers))
    realMultipliers[:,4::] = logMultipliers[:,4::] # transition probability shifts unchanged
    
    logBase = np.array([15.258112,0.259061,15.661007,0.252174])
    realBase = np.zeros(np.shape(logBase))
    realBase[0] = np.exp(logBase[0]+0.5*logBase[1]**2) # real space mu0
    realBase[2] = np.exp(logBase[2]+0.5*logBase[3]**2) # real space mu1
    realBase[1] = np.sqrt(realBase[0]**2 * (np.exp(logBase[1])-1)) # real space sigma0
    realBase[3] = np.sqrt(realBase[2]**2 * (np.exp(logBase[3])-1)) # real space sigma1
    
    logParams = convertMultToParams(logMultipliers)
    realParams = np.zeros(np.shape(logParams))
    realParams[:,0] = np.exp(logParams[:,0]+0.5*logParams[:,1]**2)
    realParams[:,2] = np.exp(logParams[:,2]+0.5*logParams[:,3]**2)
    realParams[:,1] = np.sqrt(realParams[:,0]**2 * (np.exp(logParams[:,1])-1))
    realParams[:,3] = np.sqrt(realParams[:,2]**2 * (np.exp(logParams[:,3])-1))
    
    realMultipliers[:,0:4] = realParams[:,0:4]/realBase
    
    return realMultipliers

def convertMultToParams(multipliers):
    '''convert streamflow multipliers/deltas in LHsamples to HMM parameter values'''
    params = np.zeros(np.shape(multipliers))
    params[:,0] = multipliers[:,0]*15.258112 # historical dry state mean
    params[:,1] = multipliers[:,1]*0.259061 # historical dry state std dev
    params[:,2] = multipliers[:,2]*15.661007 # historical wet state mean
    params[:,3] = multipliers[:,3]*0.252174 # historical wet state std dev
    params[:,4] = multipliers[:,4] + 0.679107 # historical dry-dry transition prob
    params[:,5] = multipliers[:,5] + 0.649169 # historical wet-wet transition prob
    
    return params

def convertParamsToMult(params):
    '''convert HMM parameter values to streamflow multipliers/deltas in LHsamples '''
    multipliers = np.zeros(np.shape(params))
    multipliers[:,0] = params[:,0]/15.258112 # historical dry state mean
    multipliers[:,1] = params[:,1]/0.259061 # historical dry state std dev
    multipliers[:,2] = params[:,2]/15.661007 # historical wet state mean
    multipliers[:,3] = params[:,3]/0.252174 # historical wet state std dev
    multipliers[:,4] = params[:,4] - 0.679107 # historical dry-dry transition prob
    multipliers[:,5] = params[:,5] - 0.649169 # historical wet-wet transition prob
    
    return multipliers

def getSamples(design, params_no, param_bounds):
    samples = np.loadtxt('../../Qgen/' + design + '.txt')[:,7:13]
    # find SOWs where mu0 > mu1 and swap their wet and dry state parameters
    HMMparams = convertMultToParams(samples)
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
            samples[i,:] = convertParamsToMult(newParams)
            
    # remove samples no longer in param_bounds
    rows_to_keep = np.union1d(np.where(samples[:,0]>=0)[0],np.where(samples[:,0]<=0)[0])
    for i in range(params_no):
        within_rows = np.intersect1d(np.where(samples[:,i] >= param_bounds[i][0])[0], np.where(samples[:,i] <= param_bounds[i][1])[0])
        rows_to_keep = np.intersect1d(rows_to_keep,within_rows)
    
    #samples = samples[rows_to_keep,:]
    
    return samples, rows_to_keep

def setupProblem(design):
    # set up problem for SALib
    if design == 'LHsamples_original_1000':
        param_bounds = np.loadtxt('../../Qgen/uncertain_params_original.txt', usecols=(1,2))[7:13,:]
    elif design == 'LHsamples_wider_1000':
        param_bounds = np.loadtxt('../../Qgen/uncertain_params_wider.txt', usecols=(1,2))[7:13,:]
    elif design == 'LHsamples_narrowed_1000':
        param_bounds = np.loadtxt('../../Qgen/uncertain_params_narrowed.txt', usecols=(1,2))[7:13,:]
     
    param_names=['mu0','sigma0','mu1','sigma1','p00','p11']
    params_no = len(param_names)
    problem = {
    'num_vars': params_no,
    'names': param_names,
    'bounds': param_bounds.tolist()
    }
    
    return param_bounds, param_names, params_no, problem

param_bounds, param_names, params_no, problem = setupProblem(design)
samples, rows_to_keep = getSamples(design, params_no, param_bounds)
RealSpaceSamples = convertLogMultToRealMult(samples)
SOWstokeep=[]
for row in rows_to_keep:
    for i in range(10):
        SOWstokeep.append(row*10+i)
LHsamples[:,7:13]=RealSpaceSamples

realizations = 10
param_bounds=np.loadtxt('../../Qgen/uncertain_params_'+design[10:-5]+'.txt', usecols=(1,2))
for i in range(7,11):
    param_bounds[i]=[round(min(LHsamples[:,i]),2),round(max(LHsamples[:,i]),2)]
SOW_values = np.array([1,1,1,1,0,0,1,1,1,1,1,0,0,0]) #Default parameter values for base SOW
params_no = len(LHsamples[0,:])
param_names=['IWRmultiplier','RESloss','TBDmultiplier','M_Imultiplier',
             'Shoshone','ENVflows','EVAdelta','XBM_mu0','XBM_sigma0',
             'XBM_mu1','XBM_sigma1','XBM_p00','XBM_p11', 'shift']


all_IDs = ['7000550','7200799','3704614']
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

def plotfailureheatmap(ID):                 
#    data= np.loadtxt('../../'+design+'/Infofiles/' +  ID + '/' + ID + '_info_0.txt')
#    
#    percentSOWs = np.zeros([len(frequencies), len(magnitudes)])
#    allSOWs = np.zeros([len(frequencies), len(magnitudes), len(LHsamples[:,0])*realizations])
#    
#    shortages = np.zeros([nMonths,len(LHsamples[:,0])*realizations])
#    demands = np.zeros([nMonths,len(LHsamples[:,0])*realizations])
#    for j in range(len(LHsamples[:,0])):
#        data= np.loadtxt('../../'+design+'/Infofiles/' +  ID + '/' + ID + '_info_' + str(j+1) + '.txt')
#        try:
#            demands[:,j*realizations:j*realizations+realizations]=data[:,idx_demand]
#            shortages[:,j*realizations:j*realizations+realizations]=data[:,idx_shortage]
#        except:
#            print('problem with ' + ID + '_info_' + str(j+1))
#            
#    #Reshape into water years
#    #Create matrix of [no. years x no. months x no. experiments]
#    f_shortages = np.zeros([int(nMonths/n),n,len(LHsamples[:,0])*realizations])
#    f_demands = np.zeros([int(nMonths/n),n,len(LHsamples[:,0])*realizations]) 
#    for i in range(len(LHsamples[:,0])*realizations):
#        f_shortages[:,:,i]= np.reshape(shortages[:,i], (int(np.size(shortages[:,i])/n), n))
#        f_demands[:,:,i]= np.reshape(demands[:,i], (int(np.size(demands[:,i])/n), n))
#    
#    # Shortage per water year
#    f_demands_WY = np.sum(f_demands,axis=1)
#    f_shortages_WY = np.sum(f_shortages,axis=1)
#    for j in range(len(frequencies)):
#        for h in range(len(magnitudes)):
#            success = np.zeros(len(LHsamples[:,0])*realizations)
#            for k in range(len(success)):
#                ratio = (f_shortages_WY[:,k]*100)/f_demands_WY[:,k]
#                if scipy.stats.percentileofscore(ratio, magnitudes[h], kind='strict')>(100-frequencies[j]):
#                    success[k]=100
#            allSOWs[j,h,:]=success
#            percentSOWs[j,h]=np.mean(success)
#    np.save('../../'+design+'/Factor_mapping/'+ ID + '_heatmap.npy',allSOWs)
    allSOWs = np.load(ID + '_heatmap.npy')#'../../'+design+'/Factor_mapping/'+ ID + '_heatmap.npy')
    return(allSOWs)       

def fitLogit(dta, predictors):
    # concatenate intercept column of 1s
    dta['Intercept'] = np.ones(np.shape(dta)[0]) 
    # get columns of predictors
    cols = dta.columns.tolist()[-1:] + predictors + ['Interaction']
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
    grid = np.column_stack([np.ones(len(x)),x,y,x*y])
 
    z = result.predict(grid)
    Z = np.reshape(z, np.shape(X))

    contourset = ax.contourf(X, Y, Z, levels, cmap=contour_cmap, aspect='auto')
    xpoints = np.mean(dta[xvar].values.reshape(-1, 10), axis=1)
    ypoints = np.mean(dta[yvar].values.reshape(-1, 10), axis=1)
    colors = np.round(np.mean(dta['Success'].values.reshape(-1, 10), axis=1),0)
    ax.scatter(xpoints, ypoints, c=colors, edgecolor='none', cmap=dot_cmap)
    ax.set_xlim(np.min(xgrid),np.max(xgrid))
    ax.set_ylim(np.min(ygrid),np.max(ygrid))
    ax.set_xlabel(xvar,fontsize=14)
    ax.set_ylabel(yvar,fontsize=14)
    ax.tick_params(axis='both',labelsize=12)
    return contourset

'''
Plot SOW points and LR contours
'''
fig, axes = plt.subplots(3,2, figsize=(18,12))
freq = [7,1,7,0,2,0]
mag = [0,7,0,3,6,7]
heatmaps = [plotfailureheatmap(all_IDs[i])/100 for i in range(len(all_IDs))]
scores = [pd.read_csv(all_IDs[i] + '_pseudo_r_scores.csv', sep=",") for i in range(len(all_IDs))]#'../../'+design+'/Factor_mapping/'+ all_IDs[i] + '_pseudo_r_scores.csv', sep=",") for i in range(len(all_IDs))]
for i in range(len(axes.flat)):
    ax = axes.flat[i]
    allSOWsperformance = heatmaps[int(i/2)]
    allSOWsperformance = allSOWsperformance[:,:,SOWstokeep]
    all_pseudo_r_scores = scores[int(i/2)]
    dta = pd.DataFrame(data = np.repeat(LHsamples[rows_to_keep,:], realizations, axis = 0), columns=param_names)
    dta['Success']=allSOWsperformance[freq[i],mag[i],:]
    pseudo_r_scores=all_pseudo_r_scores[str(frequencies[freq[i]])+'yrs_'+str(magnitudes[mag[i]])+'prc'].values
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
                      param_bounds[top_predictors[0]][1], np.around((ranges[0][1]-ranges[0][0])/500,decimals=4))
    ygrid = np.arange(param_bounds[top_predictors[1]][0], 
                      param_bounds[top_predictors[1]][1], np.around((ranges[1][1]-ranges[1][0])/500,decimals=4))
    all_predictors = [ dta.columns.tolist()[i] for i in top_predictors]
    dta['Interaction'] = dta[all_predictors[0]]*dta[all_predictors[1]]
    result = fitLogit(dta, [all_predictors[i] for i in [0,1]])  
    contourset = plotContourMap(ax, result, dta, contour_cmap, 
                                dot_cmap, contour_levels, xgrid, 
                                ygrid, all_predictors[0], all_predictors[1], base)
fig.tight_layout()
fig.savefig('./Paper1_figures/Figure_7_'+design+'.svg')
plt.close()


