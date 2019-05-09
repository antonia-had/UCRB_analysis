import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt 
plt.ioff()

LHsamples = np.loadtxt('./LHsamples.txt') 
param_bounds=np.loadtxt('./uncertain_params.txt', usecols=(1,2))
SOW_values = np.array([1,1,1,1,0,0,1,1,1,1,1,0,0]) #Default parameter values for base SOW
params_no = len(LHsamples[0,:])
param_names=['IWRmultiplier','RESloss','TBDmultiplier','M_Imultiplier',
             'Shoshone','ENVflows','EVAdelta','XBM_mu0','XBM_sigma0',
             'XBM_mu1','XBM_sigma1','XBM_p00','XBM_p11']
parameter_ranges = [[0.5, 1.5],[0.8, 1.0],[0.5, 1.5],[0.5, 1.5],[0.0, 1.0],
                    [0.0, 1.0],[-0.5, 1.0],[0.98, 1.02],[0.75, 1.25],
                    [0.98, 1.02],[0.75, 1.25],[-0.3, 0.3],[-0.3, 0.3]]
problem = {
    'num_vars': params_no,
    'names': param_names,
    'bounds': param_bounds.tolist()
}

WDs = ['36','37','38','39','45','50','51','52','53','70','72'] 

irrigation_structures = [[]]*len(WDs) 
for i in range(len(WDs)):
    irrigation_structures[i] = np.genfromtxt(WDs[i]+'_irrigation.txt',dtype='str').tolist()

irrigation_structures_flat = [item for sublist in irrigation_structures for item in sublist]
nStructures = len(irrigation_structures_flat)

# deal with fact that calling result.summary() in statsmodels.api
# calls scipy.stats.chisqprob, which no longer exists
scipy.stats.chisqprob = lambda chisq, df: scipy.stats.chi2.sf(chisq, df)

n=12 # Number of months in a year for reshaping data
nMonths = n * 105 #Record is 105 years long

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

def sensitivity_analysis_per_structure(ID):
    pseudo_r_scores = np.zeros(params_no)
    shortages = np.zeros([nMonths,len(LHsamples[:,0])])
    demands = np.zeros([nMonths,len(LHsamples[:,0])])
    for j in range(len(LHsamples[:,0])-1):
        data= np.loadtxt('./Infofiles/' +  ID + '/' + ID + '_info_' + str(j+1) + '.txt')[:,[1,2]]     
        demands[:,j]=data[:,0]
        shortages[:,j]=data[:,1]
    #Reshape into water years
    #Create matrix of [no. years x no. months x no. experiments]
    f_shortages = np.zeros([int(nMonths/n),n,len(LHsamples[:,0])])
    f_demands = np.zeros([int(nMonths/n),n,len(LHsamples[:,0])]) 
    for i in range(len(LHsamples[:,0])):
        f_shortages[:,:,i]= np.reshape(shortages[:,i], (int(np.size(shortages[:,i])/n), n))
        f_demands[:,:,i]= np.reshape(demands[:,i], (int(np.size(demands[:,i])/n), n))
    
    # Shortage per water year
    f_demands_WY = np.sum(f_demands,axis=1)
    f_shortages_WY = np.sum(f_shortages,axis=1)
    
    # Logistic regression analysis
    dta = pd.DataFrame(data = LHsamples, columns=param_names)
    success = np.ones(len(LHsamples[:,0]))
    for k in range(len(LHsamples[:,0])-1):
        ratio = 1-(f_shortages_WY[:,k]/f_demands_WY[:,k])
        if np.percentile(ratio, 30)<0.6 or np.percentile(ratio, 20)<0.55 or \
        np.percentile(ratio, 10)<0.45 or np.percentile(ratio, 5)<0.3:
            success[k]=0
    dta['Success']=success
    for m in range(params_no):
        predictors = dta.columns.tolist()[m:(m+1)]
        try:
            result = fitLogit(dta, predictors)
            pseudo_r_scores[m]=result.prsquared
        except: 
            pseudo_r_scores[m]=pseudo_r_scores[m]
    #pseudo_r_scores.to_csv('./Factor_mapping/'+ ID + '_pseudo_R2.csv')
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
    xgrid = np.arange(parameter_ranges[top_predictors[0]][0], parameter_ranges[top_predictors[0]][1], 0.01)
    ygrid = np.arange(parameter_ranges[top_predictors[1]][0], parameter_ranges[top_predictors[1]][1], 0.01)
    zgrid = np.arange(parameter_ranges[top_predictors[2]][0], parameter_ranges[top_predictors[2]][1], 0.01)
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
    fig.savefig('./Factor_mapping/'+ID+'_LR_probability.png')
    plt.close()

# Run simulation
for i in range(nStructures):
    sensitivity_analysis_per_structure(irrigation_structures_flat[i])