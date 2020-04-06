from hmmlearn.hmm import GaussianHMM
from scipy import stats as ss
import numpy as np

def fitHMM(TransformedQ):
    
    # fit HMM
    model = GaussianHMM(n_components=2, n_iter=1000).fit(np.reshape(TransformedQ,[len(TransformedQ),1]))
    hidden_states = model.predict(np.reshape(TransformedQ,[len(TransformedQ),1]))
    mus = np.array(model.means_)
    sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]),np.diag(model.covars_[1])])))
    P = np.array(model.transmat_)
    
    # re-organize mus, sigmas and P so that first row is lower mean (if not already)
    if mus[0] > mus[1]:
        mus = np.flipud(mus)
        sigmas = np.flipud(sigmas)
        P = np.fliplr(np.flipud(P))
        hidden_states = 1 - hidden_states
    
    return hidden_states, mus, sigmas, P

def findQuantiles(mus, sigmas, piNew):
    x = np.empty([10000])
    qx = np.empty([10000])
    x = np.linspace(mus[0]-4*sigmas[0], mus[1]+4*sigmas[1], 10000)
    qx = piNew[0]*ss.norm.cdf(x,mus[0],sigmas[0]) + \
        piNew[1]*ss.norm.cdf(x,mus[1],sigmas[1])
        
    return x, qx