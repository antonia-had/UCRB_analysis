import numpy as np
import statsmodels.api as sm
from hmmlearn.hmm import GaussianHMM

def fitIWRmodel(AnnualQ, AnnualIWR):
    IWRsums = np.sum(AnnualIWR,1)
    Qsums = AnnualQ[:,-1]

    Qsums_prime = Qsums - np.mean(Qsums)
    IWRsums_prime = IWRsums - np.mean(IWRsums)
    
    # fit model of IWR anomalies as function of Q anomalies
    # (no intercept b/c using anomalies)
    X = np.reshape(Qsums_prime,[len(Qsums_prime),1])
    y = IWRsums_prime
    model = sm.OLS(y,X).fit()
    
    # find mean and st dev of residuals, which are normally distributed
    mu = np.mean(model.resid)
    sigma = np.std(model.resid)
    
    return model.params, mu, sigma

def fitHMM(logAnnualQ_cut):
    # initialize matrices to store moments, transition probabilities, 
    # stationary distribution and quantiles of Gaussian HMM for each site
    nSites = np.shape(logAnnualQ_cut)[1]
    mus = np.zeros([2,nSites])
    sigmas = np.zeros([2,nSites])
    P = np.zeros([2,2,nSites])
    pi = np.zeros([2,nSites])

    for i in range(np.shape(logAnnualQ_cut)[1]):
        # fit to last 2/3 of historical record
        hmm_model = GaussianHMM(n_components=2, n_iter=1000).fit(np.reshape(logAnnualQ_cut[35::,i],[len(logAnnualQ_cut[35::,i]),1]))
        
        # find means (mus) and standard deviations (sigmas) of Gaussian mixture distributions
        mus[:,i] = np.reshape(hmm_model.means_,hmm_model.means_.size)
        sigmas[:,i] = np.reshape(np.sqrt(np.array([np.diag(hmm_model.covars_[0]),np.diag(hmm_model.covars_[1])])),hmm_model.means_.size)
        
        # find transition probabilities, P
        P[:,:,i] = hmm_model.transmat_
        
        if mus[0,i] > mus[1,i]:
            mus[:,i] = np.flipud(mus[:,i])
            sigmas[:,i] = np.flipud(sigmas[:,i])
            P[:,:,i] = np.fliplr(np.flipud(P[:,:,i]))
        
        # find stationary distribution, pi
        eigenvals, eigenvecs = np.linalg.eig(np.transpose(P[:,:,i]))
        one_eigval = np.argmin(np.abs(eigenvals-1))
        pi[:,i] = eigenvecs[:,one_eigval] / np.sum(eigenvecs[:,one_eigval])
    
    return mus, sigmas, P, pi