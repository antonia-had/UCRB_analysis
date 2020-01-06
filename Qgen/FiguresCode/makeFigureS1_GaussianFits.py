from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats as ss
import utils

def combinedQQplot(TransformedQ, mus, sigmas, P):
    
    # calculate stationary distribution
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(P))
    one_eigval = np.argmin(np.abs(eigenvals-1))
    pi = eigenvecs[:,one_eigval] / np.sum(eigenvecs[:,one_eigval])
    
    x_sorted = np.sort(TransformedQ)
    Fx_empirical = np.arange(1,len(TransformedQ)+1,1)/(len(TransformedQ)+1)
    
    x_dist, qx_dist = utils.findQuantiles(mus, sigmas, pi)
    x_fitted = np.zeros(len(x_sorted))
    for i in range(len(x_fitted)):
        x_fitted[i] = x_dist[(np.abs(qx_dist-Fx_empirical[i])).argmin()]
        
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_sorted,x_fitted,color='b')
    ax.plot([np.min(x_sorted),np.max(x_sorted)],[np.min(x_sorted),np.max(x_sorted)],color='r')
    ax.set_xlabel('Observed Quantiles\n(log of flow in m^3)',fontsize=16)
    ax.set_ylabel('Theoretical Quantiles\n(log of flow in m^3)',fontsize=16)
    ax.tick_params(axis='both',labelsize=14)
    fig.subplots_adjust(bottom=0.2,left=0.2)
    fig.savefig('FigureS1a.pdf')
    fig.clf()
    
    return None

def combinedHistogram(TransformedQ, mus, sigmas, P):
    
    # calculate stationary distribution
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(P))
    one_eigval = np.argmin(np.abs(eigenvals-1))
    pi = eigenvecs[:,one_eigval] / np.sum(eigenvecs[:,one_eigval])
    
    x = np.linspace(mus[0]-4*sigmas[0], mus[1]+4*sigmas[1], 10000)
    fx = pi[0]*ss.norm.pdf(x,mus[0],sigmas[0]) + \
        pi[1]*ss.norm.pdf(x,mus[1],sigmas[1])
            
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(TransformedQ, color='k', alpha=0.5, density=True)
    l1, = ax.plot(x, fx, c='k', linewidth=2, label='Combined')
    ax.set_xlabel('Log of annual flow in m^3')
    ax.set_ylabel('Probability Density')
    
    fig.subplots_adjust(bottom=0.2)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=1, frameon=True, fontsize=14)
    fig.savefig('FigureS1c.pdf')
    fig.clf()
            
    return None

def assessFitByState(logQ, hidden_states, mus, sigmas, P):
    
    sns.set()
    
    masks0 = hidden_states == 0
    masks1 = hidden_states == 1
    
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    ax.hist(logQ[masks0],color='r',alpha=0.5,density=True)
    ax.hist(logQ[masks1],color='b',alpha=0.5,density=True)

    x_0 = np.linspace(mus[0]-4*sigmas[0], mus[0]+4*sigmas[0], 10000)
    fx_0 = ss.norm.pdf(x_0,mus[0],sigmas[0])
    
    x_1 = np.linspace(mus[1]-4*sigmas[1], mus[1]+4*sigmas[1], 10000)
    fx_1 = ss.norm.pdf(x_1,mus[1],sigmas[1])
       
    l1, = ax.plot(x_0, fx_0, c='r', linewidth=2, label='Dry State')
    l2, = ax.plot(x_1, fx_1, c='b', linewidth=2, label='Wet State')
    ax.set_xlabel('Log of annual flow in m^3',fontsize=14)
    ax.set_ylabel('Probability Density',fontsize=14)
    fig.subplots_adjust(bottom=0.22)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=True, fontsize=14)
    fig.savefig('FigureS1d.pdf')
    fig.clf()
    
    x0_sorted = np.sort(logQ[masks0])
    p0_observed = np.arange(1,len(x0_sorted)+1,1)/(len(x0_sorted)+1)
    x0_fitted = ss.norm.ppf(p0_observed,mus[0],sigmas[0])
    
    x1_sorted = np.sort(logQ[masks1])
    p1_observed = np.arange(1,len(x1_sorted)+1,1)/(len(x1_sorted)+1)
    x1_fitted = ss.norm.ppf(p1_observed,mus[1],sigmas[1])
    
    minimum = np.min([np.min(logQ),np.min(x0_fitted),np.min(x1_fitted)])
    maximum = np.max([np.max(logQ),np.max(x0_fitted),np.max(x1_fitted)])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    p0 = ax.scatter(x0_sorted,x0_fitted,c='r')
    p1 = ax.scatter(x1_sorted,x1_fitted,c='b')
    ax.plot([minimum,maximum],[minimum,maximum],c='k')
    ax.set_xlabel('Observed Quantiles\n(log of flow in m^3)',fontsize=16)
    ax.set_ylabel('Theoretical Quantiles\n(log of flow in m^3)',fontsize=16)
    ax.tick_params(axis='both',labelsize=14)
    fig.subplots_adjust(bottom=0.2,left=0.2)
    ax.legend([p0,p1],['Dry State','Wet State'],loc='upper left',fontsize=14,numpoints=1)
    fig.savefig('FigureS1b.pdf')
    fig.clf()
    
    return None

AnnualQ = np.array(pd.read_csv('../AnnualQ.csv'))*1233.48 # convert to m^3
logQ = np.log(AnnualQ[35::,-1]) # last 70 years of log-space flows at last node
hidden_states, mus, sigmas, P = utils.fitHMM(logQ) # fit HMM

# plot Gaussian fits
combinedQQplot(logQ, mus, sigmas, P)
combinedHistogram(logQ, mus, sigmas, P)
assessFitByState(logQ, hidden_states, mus, sigmas, P)