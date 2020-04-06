from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
import utils

def plotTimeSeries(TransformedQ, hidden_states):
    
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
    ax.set_ylabel('Log of annual flow in m^3')
    
    fig.subplots_adjust(bottom=0.2)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)
    fig.savefig('FigureS2a.pdf')
    fig.clf()
    
    return None

def plotStateCorrelation(TransformedQ, hidden_states, P):
    
    sns.set()
    
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    sm.graphics.tsa.plot_acf(hidden_states,ax=ax)
    ax.set_xlim([0,10])
    
    ax = fig.add_subplot(2,1,2)
    sm.graphics.tsa.plot_pacf(hidden_states,ax=ax)
    ax.set_xlim([0,10])
    ax.set_ylim([-1,1])
    ax.set_xlabel('Lag (years)')
    
    fig.set_size_inches([9,7.25])
    fig.savefig('FigureS2b.pdf')
    fig.clf()
        
    return None

AnnualQ = np.array(pd.read_csv('../AnnualQ.csv'))*1233.48 # convert to m^3
logQ = np.log(AnnualQ[35::,-1]) # last 70 years of log-space flows at last node
hidden_states, mus, sigmas, P = utils.fitHMM(logQ) # fit HMM

plotTimeSeries(logQ, hidden_states)
plotStateCorrelation(logQ, hidden_states, P)