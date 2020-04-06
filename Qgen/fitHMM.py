from hmmlearn.hmm import GaussianHMM
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats as ss

def fitHMM(TransformedQ):
    
    # fit HMM to last 2/3 of data, validate on 1st 3rd
    model = GaussianHMM(n_components=2, n_iter=1000).fit(np.reshape(TransformedQ[35::],[len(TransformedQ[35::]),1]))
    hidden_states = model.predict(np.reshape(TransformedQ,[len(TransformedQ),1]))
    mus = np.array(model.means_)
    sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]),np.diag(model.covars_[1])])))
    P = np.array(model.transmat_)
    
    #logProb = model.score(np.reshape(TransformedQ,[len(TransformedQ),1]))
    #samples = model.sample(105)
    
    # re-organize mus, sigmas and P so that first row is lower mean (if not already)
    if mus[0] > mus[1]:
        mus = np.flipud(mus)
        sigmas = np.flipud(sigmas)
        P = np.fliplr(np.flipud(P))
        hidden_states = 1 - hidden_states
    
    return hidden_states, mus, sigmas, P

def plotTimeSeries(TransformedQ, hidden_states, ylabel, filename):
    
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    xs = np.arange(len(TransformedQ))+1909
    masks = hidden_states == 0
    ax.scatter(xs[masks], TransformedQ[masks], c='r', label='Dry State')
    masks = hidden_states == 1
    ax.scatter(xs[masks], TransformedQ[masks], c='b', label='Wet State')
    ax.plot(xs, TransformedQ, c='k')
    
    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    
    fig.subplots_adjust(bottom=0.2)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)
    fig.savefig(filename)
    fig.clf()
    
    return None

def plotDistribution(TransformedQ, mus, sigmas, P):
    
    # calculate stationary distribution
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(P))
    one_eigval = np.argmin(np.abs(eigenvals-1))
    pi = eigenvecs[:,one_eigval] / np.sum(eigenvecs[:,one_eigval])
    
    x_0 = np.linspace(mus[0]-4*sigmas[0], mus[0]+4*sigmas[0], 10000)
    fx_0 = pi[0]*ss.norm.pdf(x_0,mus[0],sigmas[0])
    
    x_1 = np.linspace(mus[1]-4*sigmas[1], mus[1]+4*sigmas[1], 10000)
    fx_1 = pi[1]*ss.norm.pdf(x_1,mus[1],sigmas[1])
    
    x = np.linspace(mus[0]-4*sigmas[0], mus[1]+4*sigmas[1], 10000)
    fx = pi[0]*ss.norm.pdf(x,mus[0],sigmas[0]) + \
        pi[1]*ss.norm.pdf(x,mus[1],sigmas[1])
            
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(TransformedQ, color='k', alpha=0.5, density=True)
    l1, = ax.plot(x_0, fx_0, c='r', linewidth=2, label='Dry State Distn')
    l2, = ax.plot(x_1, fx_1, c='b', linewidth=2, label='Wet State Distn')
    l3, = ax.plot(x, fx, c='k', linewidth=2, label='Combined State Distn')
    
    fig.subplots_adjust(bottom=0.15)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=True)
    fig.savefig('MixedGaussianFit.png')
    fig.clf()
            
    return None

plt.switch_backend('agg')

AnnualQ = np.array(pd.DataFrame.from_csv('AnnualQ.csv'))

logQ = np.log(AnnualQ[:,-1])
hidden_states, mus, sigmas, P = fitHMM(logQ)
plotDistribution(logQ, mus, sigmas, P)
plotTimeSeries(logQ, hidden_states, 'log(Flow at Site 208)', 'StateTseries_Log.png')
print(mus)
print(P)
#print(samples)
