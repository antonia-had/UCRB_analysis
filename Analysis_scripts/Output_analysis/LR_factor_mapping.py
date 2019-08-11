import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt 
plt.switch_backend('agg')
import itertools
from mpi4py import MPI
import math
plt.ioff()

LHsamples = np.loadtxt('./Global_experiment_uncurtailed/LHsamples_wider.txt') 
realizations = 10
param_bounds=np.loadtxt('./Global_experiment_uncurtailed/uncertain_params_wider.txt', usecols=(1,2))
SOW_values = np.array([1,1,1,1,0,0,1,1,1,1,1,0,0,0]) #Default parameter values for base SOW
params_no = len(LHsamples[0,:])
param_names=['IWRmultiplier','RESloss','TBDmultiplier','M_Imultiplier',
             'Shoshone','ENVflows','EVAdelta','XBM_mu0','XBM_sigma0',
             'XBM_mu1','XBM_sigma1','XBM_p00','XBM_p11', 'shift']


irrigation_structures = np.genfromtxt('./Global_experiment_uncurtailed/irrigation.txt',dtype='str').tolist()
TBDs = np.genfromtxt('./Global_experiment_uncurtailed/TBD.txt',dtype='str').tolist()
all_IDs = np.genfromtxt('./Global_experiment_uncurtailed/metrics_structures.txt',dtype='str').tolist()[:24]
nStructures = len(all_IDs)

# deal with fact that calling result.summary() in statsmodels.api
# calls scipy.stats.chisqprob, which no longer exists
scipy.stats.chisqprob = lambda chisq, df: scipy.stats.chi2.sf(chisq, df)

n=12 # Number of months in a year for reshaping data
nMonths = n * 105 #Record is 105 years long

idx_shortage = np.arange(2,22,2)
idx_demand = np.arange(1,21,2)

def fitLogit(dta, predictors):
    # concatenate intercept column of 1s
    dta['Intercept'] = np.ones(np.shape(dta)[0]) 
    # get columns of predictors
    cols = dta.columns.tolist()[-1:] + predictors 
    #fit logistic regression
    logit = sm.Logit(dta['Success'], dta[cols])
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

def shortage_duration(sequence, value):
    cnt_shrt = [sequence[i]>value for i in range(len(sequence))] # Returns a list of True values when there's a shortage about the value
    shrt_dur = [ sum( 1 for _ in group ) for key, group in itertools.groupby( cnt_shrt ) if key ] # Counts groups of True values
    return shrt_dur

def factor_mapping(ID):
    print(ID)
    if ID!='7202003':
        shortages = np.zeros([nMonths,len(LHsamples[:,0])*realizations])
        demands = np.zeros([nMonths,len(LHsamples[:,0])*realizations])
        for j in range(len(LHsamples[:,0])):
            data= np.loadtxt('./Global_experiment_uncurtailed/Infofiles_wide/' +  ID + '/' + ID + '_info_' + str(j+1) + '.txt')
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
        
        if ID in irrigation_structures:
            print(ID + ' is an irrigation structure')
            fail_duration = [20, 10, 5, 2]
            fail_shortage = [20, 30, 50, 70]
            for j in range(len(fail_duration)):
                for h in range(len(fail_shortage)):
                    pseudo_r_scores = np.zeros(params_no)
                    # Logistic regression analysis
                    dta = pd.DataFrame(data = np.repeat(LHsamples, realizations, axis = 0), columns=param_names)
                    success = np.ones(len(LHsamples[:,0])*realizations)
                    for k in range(len(success)):
                        # Time series of ratio of shortage to demand
                        ratio = (f_shortages_WY[:,k]*100)/f_demands_WY[:,k]
                        if np.percentile(ratio, fail_duration[j])>fail_shortage[h]:
                            success[k]=0
                    dta['Success']=success
                    for m in range(params_no):
                        predictors = dta.columns.tolist()[m:(m+1)]
                        try:
                            result = fitLogit(dta, predictors)
                            pseudo_r_scores[m]=result.prsquared
                        except: 
                            pseudo_r_scores[m]=pseudo_r_scores[m]
                    print('got logit results')
                    np.savetxt('./Global_experiment_uncurtailed/Factor_mapping/'+ ID +'_'+str(fail_duration[j])+'yrsw'+str(fail_shortage[h])+\
                                   '.csv', pseudo_r_scores, delimiter=",")
                    if pseudo_r_scores.any():
                        fig, axes = plt.subplots(1,3)
                        axes = axes.ravel()
                        top_predictors = np.argsort(pseudo_r_scores)[::-1][:3] #Sort scores and pick top 3 predictors
                          
                        # define color map for dots representing SOWs in which the policy
                        # succeeds (light blue) and fails (dark red)
                        dot_cmap = mpl.colors.ListedColormap(np.array([[227,26,28],[166,206,227]])/255.0)
                         
                        # define color map for probability contours
                        contour_cmap = mpl.cm.get_cmap('RdBu')
                         
                        # define probability contours
                        contour_levels = np.arange(0.0, 1.05,0.1)
                        
                        # define base values of the predictors
                        base = SOW_values[top_predictors]
                         
                        # define grid of x (1st predictor), and y (2nd predictor) dimensions
                        # to plot contour map over
                        xgrid = np.arange(param_bounds[top_predictors[0]][0], param_bounds[top_predictors[0]][1], 0.01)
                        ygrid = np.arange(param_bounds[top_predictors[1]][0], param_bounds[top_predictors[1]][1], 0.01)
                        zgrid = np.arange(param_bounds[top_predictors[2]][0], param_bounds[top_predictors[2]][1], 0.01)
                        all_predictors = [ dta.columns.tolist()[i] for i in top_predictors]
                        
                        #Axes 0
                        result = fitLogit(dta, [all_predictors[i] for i in [0,1]])
                        # plot contour map when 3rd predictor ('x3') is held constant
                        contourset = plotContourMap(axes[0], result, dta, contour_cmap, dot_cmap, contour_levels, xgrid, ygrid, all_predictors[0], all_predictors[1], base)
                        
                        #Axes 1
                        result = fitLogit(dta, [all_predictors[i] for i in [0,2]])
                        # plot contour map when 3rd predictor ('x3') is held constant
                        contourset = plotContourMap(axes[1], result, dta, contour_cmap, dot_cmap, contour_levels, xgrid, zgrid, all_predictors[0], all_predictors[2], base)    
                        
                        #Axes 2
                        result = fitLogit(dta, [all_predictors[i] for i in [1,2]])
                        # plot contour map when 3rd predictor ('x3') is held constant
                        contourset = plotContourMap(axes[2], result, dta, contour_cmap, dot_cmap, contour_levels, ygrid, zgrid, all_predictors[1], all_predictors[2], base) 
                        
                        plt.show()     
                        fig.subplots_adjust(wspace=0.5,hspace=0.3,right=0.8)
                        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                        cbar = fig.colorbar(contourset, cax=cbar_ax)
                        cbar_ax.set_ylabel('Probability',fontsize=12)
                        yticklabels = cbar.ax.get_yticklabels()
                        cbar.ax.set_yticklabels(yticklabels,fontsize=10)
                        fig.set_size_inches([14.5,8])
                        fig.suptitle('Probability of not exceeding historic shortage each percentile for '+ID)
                        #fig.savefig('./Factor_mapping/'+ID+'_LR_probability.svg')
                        fig.savefig('./Global_experiment_uncurtailed/Factor_mapping/Figures/'+\
                                    ID+'_'+str(fail_duration[j])+'yrsw'+str(fail_shortage[h])+\
                                    'pcshort.png')
                        plt.close()
        
        else:
            print(ID + ' is not an irrigation structure')
            fail_duration = [10, 7, 5]
            fail_shortage = [5, 10, 20]
            for j in range(len(fail_duration)):
                for h in range(len(fail_shortage)):
                    pseudo_r_scores = np.zeros(params_no)
                    # Logistic regression analysis
                    dta = pd.DataFrame(data = np.repeat(LHsamples, realizations, axis = 0), columns=param_names)
                    success = np.ones(len(LHsamples[:,0])*realizations)
                    for k in range(len(success)):
                        # Time series of ratio of shortage to demand
                        ratio = (f_shortages_WY[:,k]*100)/f_demands_WY[:,k]
                        durations = shortage_duration(ratio, fail_shortage[j])
                        if len(durations)>0:
                            if max(durations)>fail_duration[h]:
                                success[k]=0
                    dta['Success']=success
                    for m in range(params_no):
                        predictors = dta.columns.tolist()[m:(m+1)]
                        try:
                            result = fitLogit(dta, predictors)
                            pseudo_r_scores[m]=result.prsquared
                        except: 
                            pseudo_r_scores[m]=pseudo_r_scores[m]
                    print('got logit results')
                    np.savetxt('./Global_experiment_uncurtailed/Factor_mapping/'+ ID +'_'+str(fail_duration[j])+'yrsw'+str(fail_shortage[h])+\
                                   '.csv', pseudo_r_scores, delimiter=",")
                    if pseudo_r_scores.any():
                        fig, axes = plt.subplots(1,3)
                        axes = axes.ravel()
                        top_predictors = np.argsort(pseudo_r_scores)[::-1][:3] #Sort scores and pick top 3 predictors
                          
                        # define color map for dots representing SOWs in which the policy
                        # succeeds (light blue) and fails (dark red)
                        dot_cmap = mpl.colors.ListedColormap(np.array([[227,26,28],[166,206,227]])/255.0)
                         
                        # define color map for probability contours
                        contour_cmap = mpl.cm.get_cmap('RdBu')
                         
                        # define probability contours
                        contour_levels = np.arange(0.0, 1.05,0.1)
                        
                        # define base values of the predictors
                        base = SOW_values[top_predictors]
                         
                        # define grid of x (1st predictor), and y (2nd predictor) dimensions
                        # to plot contour map over
                        xgrid = np.arange(param_bounds[top_predictors[0]][0], param_bounds[top_predictors[0]][1], 0.01)
                        ygrid = np.arange(param_bounds[top_predictors[1]][0], param_bounds[top_predictors[1]][1], 0.01)
                        zgrid = np.arange(param_bounds[top_predictors[2]][0], param_bounds[top_predictors[2]][1], 0.01)
                        all_predictors = [ dta.columns.tolist()[i] for i in top_predictors]
                        
                        #Axes 0
                        result = fitLogit(dta, [all_predictors[i] for i in [0,1]])
                        # plot contour map when 3rd predictor ('x3') is held constant
                        contourset = plotContourMap(axes[0], result, dta, contour_cmap, dot_cmap, contour_levels, xgrid, ygrid, all_predictors[0], all_predictors[1], base)
                        
                        #Axes 1
                        result = fitLogit(dta, [all_predictors[i] for i in [0,2]])
                        # plot contour map when 3rd predictor ('x3') is held constant
                        contourset = plotContourMap(axes[1], result, dta, contour_cmap, dot_cmap, contour_levels, xgrid, zgrid, all_predictors[0], all_predictors[2], base)    
                        
                        #Axes 2
                        result = fitLogit(dta, [all_predictors[i] for i in [1,2]])
                        # plot contour map when 3rd predictor ('x3') is held constant
                        contourset = plotContourMap(axes[2], result, dta, contour_cmap, dot_cmap, contour_levels, ygrid, zgrid, all_predictors[1], all_predictors[2], base) 
                        
                        plt.show()     
                        fig.subplots_adjust(wspace=0.5,hspace=0.3,right=0.8)
                        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                        cbar = fig.colorbar(contourset, cax=cbar_ax)
                        cbar_ax.set_ylabel('Probability',fontsize=12)
                        yticklabels = cbar.ax.get_yticklabels()
                        cbar.ax.set_yticklabels(yticklabels,fontsize=10)
                        fig.set_size_inches([14.5,8])
                        fig.suptitle('Probability of not exceeding historic shortage each percentile for '+ID)
                        #fig.savefig('./Factor_mapping/'+ID+'_LR_probability.svg')
                        fig.savefig('./Global_experiment_uncurtailed/Factor_mapping/Figures/'+\
                                    ID+'_'+str(fail_duration[j])+'yrsw'+str(fail_shortage[h])+\
                                    'pcshort.png')
                        plt.close()
                        
    elif ID=='7202003':
        print(ID + ' is 15-mile reach')
        streamflow = np.zeros([nMonths,len(LHsamples[:,0])*realizations])
        streamflowhistoric = np.zeros([nMonths])
        for j in range(len(LHsamples[:,0])):
            data= np.loadtxt('./Global_experiment_uncurtailed/Infofiles_wide/' +  ID + '/' + ID + '_streamflow_' + str(j+1) + '.txt')[:,1:]
            streamflow[:,j*realizations:j*realizations+realizations]=data
        streamflowhistoric = np.loadtxt('./Global_experiment_uncurtailed/Infofiles_wide/' +  ID + '/' + ID + '_streamflow_0.txt')[:,1]
        
        fail_shortage = [35000, 40000, 48900, 55000]
        for h in range(len(fail_shortage)):
            pseudo_r_scores = np.zeros(params_no)
            # Logistic regression analysis
            dta = pd.DataFrame(data = np.repeat(LHsamples, realizations, axis = 0), columns=param_names)
            historicworst = max(shortage_duration(streamflowhistoric, fail_shortage[h]))
            success = np.ones(len(LHsamples[:,0])*realizations)
            for k in range(len(success)):
                durations = shortage_duration(streamflow[:,k], fail_shortage[h])
                if len(durations)>0:
                    if max(durations)>historicworst:
                        success[k]=0
            dta['Success']=success
            for m in range(params_no):
                predictors = dta.columns.tolist()[m:(m+1)]
                try:
                    result = fitLogit(dta, predictors)
                    pseudo_r_scores[m]=result.prsquared
                except: 
                    pseudo_r_scores[m]=pseudo_r_scores[m]
            np.savetxt('./Global_experiment_uncurtailed/Factor_mapping/'+ ID +'_'+str(fail_shortage[h])+\
                           '.csv', pseudo_r_scores, delimiter=",")
            if pseudo_r_scores.any():
                fig, axes = plt.subplots(1,3)
                axes = axes.ravel()
                top_predictors = np.argsort(pseudo_r_scores)[::-1][:3] #Sort scores and pick top 3 predictors
                  
                # define color map for dots representing SOWs in which the policy
                # succeeds (light blue) and fails (dark red)
                dot_cmap = mpl.colors.ListedColormap(np.array([[227,26,28],[166,206,227]])/255.0)
                 
                # define color map for probability contours
                contour_cmap = mpl.cm.get_cmap('RdBu')
                 
                # define probability contours
                contour_levels = np.arange(0.0, 1.05,0.1)
                
                # define base values of the predictors
                base = SOW_values[top_predictors]
                 
                # define grid of x (1st predictor), and y (2nd predictor) dimensions
                # to plot contour map over
                xgrid = np.arange(param_bounds[top_predictors[0]][0], param_bounds[top_predictors[0]][1], 0.01)
                ygrid = np.arange(param_bounds[top_predictors[1]][0], param_bounds[top_predictors[1]][1], 0.01)
                zgrid = np.arange(param_bounds[top_predictors[2]][0], param_bounds[top_predictors[2]][1], 0.01)
                all_predictors = [ dta.columns.tolist()[i] for i in top_predictors]
                
                #Axes 0
                result = fitLogit(dta, [all_predictors[i] for i in [0,1]])
                # plot contour map when 3rd predictor ('x3') is held constant
                contourset = plotContourMap(axes[0], result, dta, contour_cmap, dot_cmap, contour_levels, xgrid, ygrid, all_predictors[0], all_predictors[1], base)
                
                #Axes 1
                result = fitLogit(dta, [all_predictors[i] for i in [0,2]])
                # plot contour map when 3rd predictor ('x3') is held constant
                contourset = plotContourMap(axes[1], result, dta, contour_cmap, dot_cmap, contour_levels, xgrid, zgrid, all_predictors[0], all_predictors[2], base)    
                
                #Axes 2
                result = fitLogit(dta, [all_predictors[i] for i in [1,2]])
                # plot contour map when 3rd predictor ('x3') is held constant
                contourset = plotContourMap(axes[2], result, dta, contour_cmap, dot_cmap, contour_levels, ygrid, zgrid, all_predictors[1], all_predictors[2], base) 
                
                plt.show()     
                fig.subplots_adjust(wspace=0.5,hspace=0.3,right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                cbar = fig.colorbar(contourset, cax=cbar_ax)
                cbar_ax.set_ylabel('Probability',fontsize=12)
                yticklabels = cbar.ax.get_yticklabels()
                cbar.ax.set_yticklabels(yticklabels,fontsize=10)
                fig.set_size_inches([14.5,8])
                fig.suptitle('Probability of not exceeding historic shortage each percentile for '+ID)
                #fig.savefig('./Factor_mapping/'+ID+'_LR_probability.svg')
                fig.savefig('./Global_experiment_uncurtailed/Factor_mapping/Figures/'+\
                            ID+'_'+str(fail_duration[j])+'yrsw'+str(fail_shortage[h])+\
                            'pcshort.png')
                plt.close()
        

    
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