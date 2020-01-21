from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats as ss
#import utils

AnnualQ = np.array(pd.read_csv('../AnnualQ.csv')) # flows in acre-ft
logQ = np.log(AnnualQ[35::,-1]) # last 70 years of log-space flows at last node
#hidden_states, mus, sigmas, P = utils.fitHMM(logQ) # fit HMM
#print(hidden_states)
#print(mus)
#print(sigmas)
#print(P)
hidden_states = np.array([0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,1,0,0,0,1,1,1,1,1,1,1,0,0,1,1,1,0,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0])
mus = np.array([[15.26333283],[15.66367286]])
sigmas = np.array([[0.25903638],[0.25132753]])
P = np.array([[0.6794469,0.3205531 ],[0.34904974,0.65095026]])

uncertain_params = np.loadtxt('../uncertain_params_original.txt',usecols=[1,2],skiprows=7)
mus_mult_LB = uncertain_params[0,0]
mus_mult_UB = uncertain_params[0,1]
sigmas_mult_LB = uncertain_params[1,0]
sigmas_mult_UB = uncertain_params[1,1]

def convertMoments(mus, sigmas):
    # convert mean of log(Q in acre-ft) to mean of log(Q in m^3)
    # stdev of log(Q in acre-ft) same as stdev of log(Q in m^3)
    mus_R = np.exp(mus + 0.5*sigmas**2) # real space mean of Q in acre-ft
    mus_m3 = mus*(1 + np.log(1233.48)/(np.log(mus_R) - 0.5*sigmas) )
    
    return mus_m3

def getCombinedPDF(mus, sigmas, pi, annualQ):
    x = np.linspace(0, 1.5*np.max(annualQ), 10000)
    fx = pi[0]*ss.lognorm.pdf(x,sigmas[0],0,np.exp(mus[0])) + \
        pi[1]*ss.lognorm.pdf(x,sigmas[1],0,np.exp(mus[1]))
    
    return x, fx

def getStatePDFs(mus, sigmas, annualQ):
    x0 = np.linspace(0, 1.5*np.max(annualQ), 10000)
    fx0 = ss.lognorm.pdf(x0,sigmas[0],0,np.exp(mus[0]))
    
    x1 = np.linspace(0, 1.5*np.max(annualQ), 10000)
    fx1 = ss.lognorm.pdf(x1,sigmas[1],0,np.exp(mus[1]))
    
    return x0, fx0, x1, fx1

def combinedHistogram(annualQ, mus, sigmas, P, ax, change):
    
    # calculate stationary distribution
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(P))
    one_eigval = np.argmin(np.abs(eigenvals-1))
    pi = eigenvecs[:,one_eigval] / np.sum(eigenvecs[:,one_eigval])
    
    mus_m3 = convertMoments(mus, sigmas)
    x, fx = getCombinedPDF(mus_m3, sigmas, pi, annualQ)
            
    ax.hist(annualQ, color='k', alpha=0.5, density=True)
    l1, = ax.plot(x, fx, c='k', linewidth=2, label='Combined')
    
    if change == 'mean':
        mus_lower = mus*mus_mult_LB # change to lower bound mean for both states
        mus_m3_lower = convertMoments(mus_lower, sigmas)
        x_lower, fx_lower = getCombinedPDF(mus_m3_lower, sigmas, pi, annualQ)
        
        mus_upper = mus*mus_mult_UB # change to upper bound mean for both states
        mus_m3_upper = convertMoments(mus_upper, sigmas)
        x_upper, fx_upper = getCombinedPDF(mus_m3_upper, sigmas, pi, annualQ)
        
    elif change == 'std':
        sigmas_lower = sigmas*sigmas_mult_LB # change to lower bound stdev for both states
        mus_m3 = convertMoments(mus, sigmas_lower)
        x_lower, fx_lower = getCombinedPDF(mus_m3, sigmas_lower, pi, annualQ)
        
        sigmas_upper = sigmas*sigmas_mult_UB # change to upper bound stdev for both states
        mus_m3 = convertMoments(mus, sigmas_upper)
        x_upper, fx_upper = getCombinedPDF(mus_m3, sigmas_upper, pi, annualQ)
        
    l2, = ax.plot(x_lower, fx_lower, c='k', linewidth=2, linestyle='--')
    l3, = ax.plot(x_upper, fx_upper, c='k', linewidth=2, linestyle='--')
            
    return ax, l1

def assessFitByState(annualQ, hidden_states, mus, sigmas, P, ax, change, state):
    
    masks0 = hidden_states == 0
    masks1 = hidden_states == 1
    
    ax.hist(annualQ[masks0],color='r',alpha=0.5,density=True)
    ax.hist(annualQ[masks1],color='b',alpha=0.5,density=True)
    
    mus_m3 = convertMoments(mus, sigmas)
    x0, fx0, x1, fx1 = getStatePDFs(mus_m3, sigmas, annualQ)
       
    l0, = ax.plot(x0, fx0, c='r', linewidth=2, label='Dry State')
    l1, = ax.plot(x1, fx1, c='b', linewidth=2, label='Wet State')
    
    if change == 'mean':
        mus_lower = mus*mus_mult_LB
        mus_m3_lower = convertMoments(mus_lower, sigmas)
        x0_lower, fx0_lower, x1_lower, fx1_lower = getStatePDFs(mus_m3_lower, sigmas, annualQ)
        
        mus_upper = mus*mus_mult_UB
        mus_m3_upper = convertMoments(mus_upper, sigmas)
        x0_upper, fx0_upper, x1_upper, fx1_upper = getStatePDFs(mus_m3_upper, sigmas, annualQ)
        
    elif change == 'std':
        sigmas_lower = sigmas*sigmas_mult_LB
        mus_m3 = convertMoments(mus, sigmas_lower)
        x0_lower, fx0_lower, x1_lower, fx1_lower = getStatePDFs(mus_m3, sigmas_lower, annualQ)
        
        sigmas_upper = sigmas*sigmas_mult_UB
        mus_m3 = convertMoments(mus, sigmas_upper)
        x0_upper, fx0_upper, x1_upper, fx1_upper = getStatePDFs(mus_m3, sigmas_upper, annualQ)
        
    if state == 'dry':
        l0_lower, = ax.plot(x0_lower, fx0_lower, c='r', linewidth=2, linestyle='--')
        l0_upper, = ax.plot(x0_upper, fx0_upper, c='r', linewidth=2, linestyle='--')
    elif state == 'wet':
        l1_lower, = ax.plot(x1_lower, fx1_lower, c='b', linewidth=2, linestyle='--')
        l1_upper, = ax.plot(x1_upper, fx1_upper, c='b', linewidth=2, linestyle='--')
    
    return ax, l0, l1

AnnualQ = AnnualQ[35::,-1]*1233.48 # convert to m^3

# plot real-space fits
sns.set()
fig = plt.figure()
ax = fig.add_subplot(3,2,1)
ax, l_dry, l_wet = assessFitByState(AnnualQ, hidden_states, mus, sigmas, P, ax, 'mean', 'dry')
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.set_title('Changes in ' + r'$\mu_0$')

ax = fig.add_subplot(3,2,2)
ax, l_dry, l_wet = assessFitByState(AnnualQ, hidden_states, mus, sigmas, P, ax, 'std', 'dry')
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.set_title('Changes in ' + r'$\sigma_0$')

ax = fig.add_subplot(3,2,3)
ax, l_dry, l_wet = assessFitByState(AnnualQ, hidden_states, mus, sigmas, P, ax, 'mean', 'wet')
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.set_ylabel('Probability Density')
ax.set_title('Changes in ' + r'$\mu_1$')

ax = fig.add_subplot(3,2,4)
ax, l_dry, l_wet = assessFitByState(AnnualQ, hidden_states, mus, sigmas, P, ax, 'std', 'wet')
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.set_title('Changes in ' + r'$\sigma_1$')

ax = fig.add_subplot(3,2,5)
ax, l_combined = combinedHistogram(AnnualQ, mus, sigmas, P, ax, 'mean')
ax.set_xlabel('Annual flow in m^3')
ax.set_title('Changes in ' + r'$\mu_0$' + ' and ' + r'$\mu_1$')

ax = fig.add_subplot(3,2,6)
ax, l_combined = combinedHistogram(AnnualQ, mus, sigmas, P, ax, 'std')
ax.set_xlabel('Annual flow in m^3')
ax.set_title('Changes in ' + r'$\sigma_0$' + ' and ' + r'$\sigma_1$')

fig.subplots_adjust(bottom=0.15)
fig.legend([l_dry, l_wet, l_combined],['Dry State','Wet State','Combined'],loc='lower center',ncol=3,fontsize=14)
fig.set_size_inches([7.3,7.4])
fig.savefig('FigureS2.pdf')
fig.clf()
