
'''
##in terminal, type: python setup.py build_ext --inplace
'''

import numpy as np
cimport numpy as np
cimport cython
cimport libc.math

from scipy.stats import multivariate_normal
#better than np.random.multivariate_normal

ctypedef np.float_t float_t
ctypedef np.int_t int_t



class dtghmm:
    
    """Discrete time Gaussian hidden Markov model
    
    Usage:
        1. independent univariate/multivariate Gaussian hidden Markov model
        w/o covariates, w/o ridge regularization 
        
        2. 1st-order (vector) autoregressive hidden Markov model w/o ridge
        regularization
    
    Parameters
    ----------

    nstate: int
        Number of states in the model
    ar1: boolean
        When True, 1st-order autoregressive hidden Markov model
        When False, independent Gaussian Hidden Markov model
    prior: 1-d array
        prior state probabilities
    tpm: 2-d array
        Transition probability matrix.
    covcube: 3-d array
        covariance matrix in each latent state. The first index corresponds
        to the latent state.
    coefcube: 3-d array
        coefficient matrix in each latent state. The first index corresponds
        to the latent states. In independent Gaussian hidden Markov model, this
        is the intercept and regression slopes for covariates in each state; in
        1st-order autoregressive hidden Markov model, this is the intercept and
        autoregression coefficients in each state."""
        
    
    def __init__(self, nstate, ar1, prior, tpm, covcube, coefcube):
        self.nstate = nstate #check int >= 2
        self.ar1 = ar1   #independent or ar1
        self.prior = prior
        self.tpm = tpm
        self.covcube = covcube
        self.coefcube = coefcube
     

    def simulate(self, ns, xmat=None):
        """
        simulate the observations and states for discrete time Gaussian hidden
        Markov model w/o covariates
        
        Parameters
        ----------
        
        ns: int
            length of the times series
        xmat: 2-d array, default to None, only needed if self.ar1 == False
            design matrix for exogenous variables. For 1st-order autoregressive hidden
            Markov model, xmat should be empty; for independent Gaussian hidden 
            Markov model, the first column of xmat should be 1 for intercept and the
            rest represents the covariates if any.
        """
        if self.ar1 == True:
            y, state = sim_ar1hmm(ns, self.prior, self.tpm, 
                                  self.covcube, self.coefcube)
        else:
            y, state = sim_ghmm(ns, self.prior, self.tpm, 
                                self.covcube, self.coefcube, xmat)
    
        return y, state
    
    
    def fit(self, ymat, ntimes, xmat = None, coefshrink = 0,
            covshrink = 0, maxit = 100, tol = 1e-4):
        """
        Fit discrete time Gaussian hidden Markov model w/o covariates using EM
        algorithm w/o ridge regularization
        
        Parameters
        ----------
        ymat: 2-d array
            observed univariate/multivariate times series
        ntimes: 1-d array
            length values for each individual series
        xmat: 2-d array, default to None, only needed if self.ar1 == False.
            design matrix for exogenous variables. For 1st-order autoregressive hidden
            Markov model, xmat should be empty; for independent Gaussian hidden 
            Markov model, the first column of xmat should be 1 for intercept and the
            rest represents the covariates if any.
        coefshrink: double, default to 0
            ridge regularization parameter, which should be nonnegative.
        covshrink: double, default to 0
            regularization parameter on the covariance matrix, must be between
            0 and 1.
        maxit: int, default to 100
            maximum number of EM iterations.
        tol: double, default to 1e-4
            tolerance for the EM algorithm to stop
        """
        if self.ar1 == True:
            nllk = EM_ar1hmm(self.nstate, self.prior, self.tpm, self.covcube, 
                 self.coefcube, ntimes, ymat, 
                 coefshrink, covshrink)
            
            oldnllk = nllk
            
            for i in range(maxit):
   
                nllk = EM_ar1hmm(self.nstate, self.prior, self.tpm, self.covcube, 
                       self.coefcube, ntimes, ymat, 
                       coefshrink, covshrink)
                if i >= 1:
                    print("iteration", i+1, "; negloglik", nllk)
                    if (np.abs(nllk-oldnllk))/np.abs(oldnllk) < tol or nllk > oldnllk:
                        nllk = oldnllk
                        break
               
                oldnllk = nllk
        else:
            
            nllk = EM_dtghmm(self.nstate, self.prior, self.tpm, self.covcube, 
                 self.coefcube, ntimes, xmat, ymat, 
                 coefshrink, covshrink)
            oldnllk = nllk
            
            for i in range(maxit):
                
                nllk = EM_dtghmm(self.nstate, self.prior, self.tpm, self.covcube, 
                       self.coefcube, ntimes, xmat, ymat, 
                       coefshrink, covshrink)
                if i >= 1:
                    print("iteration", i+1, "; negloglik", nllk)
                
                    if (np.abs(nllk-oldnllk))/np.abs(oldnllk) < tol or nllk > oldnllk:
                        nllk = oldnllk
                        break
                
                oldnllk = nllk
        return nllk
    
    
    def viterbi(self, ymat, xmat=None):
        '''
        Optimal path decoding via Viterbi algorithm
        
        Parameters
        ----------
        ymat: 2-d array
            observed univariate/multivariate times series
        xmat: 2-d array, default to None, only needed if self.ar1 == False.
            design matrix for exogenous variables. For 1st-order autoregressive hidden
            Markov model, xmat should be empty; for independent Gaussian hidden 
            Markov model, the first column of xmat should be 1 for intercept and the
            rest represents the covariates if any.
        '''
        ns = ymat.shape[0]
        nodeprob = np.zeros([ns, self.nstate])
        
        if self.ar1 == True:
            nodeprob_ar1hmm(self.nstate, self.covcube, 
                  self.coefcube, ymat, nodeprob)
            
        else:
            nodeprob_dtghmm(self.nstate, self.covcube, 
                  self.coefcube, ymat, xmat, nodeprob)
            
        xi = np.zeros(nodeprob.shape)
        
        foo = self.prior * nodeprob[0,:]
        xi[0,:] = foo / np.sum(foo)
        
        for i in range(1, ns):
            mult = xi[i-1, :]
            for vv in range(1,self.nstate):
                mult = np.vstack((mult, xi[i-1,:]))
            foo = np.max(mult.T * self.tpm, axis=0) * nodeprob[i,:]
            xi[i,:] = foo / np.sum(foo)

        state = np.repeat(0,ns)
        state[ns-1] = np.argmax(xi[ns-1,:])
        
        for i in range(ns-2,-1,-1):
            state[i] = np.argmax(self.tpm[:, state[i+1]] * xi[i,:])
        
        return state
    
    
    def smooth(self, ymat, xmat=None):
        '''
        Smooth the posterior state probabilities
        
        Parameters
        ----------
        ymat: 2-d array
            observed univariate/multivariate times series
        xmat: 2-d array, default to None, only needed if self.ar1 == False.
            design matrix for exogenous variables. For 1st-order autoregressive hidden
            Markov model, xmat should be empty; for independent Gaussian hidden 
            Markov model, the first column of xmat should be 1 for intercept and the
            rest represents the covariates if any.
        '''
        
        ns = ymat.shape[0]
        nodeprob = np.zeros([ns, self.nstate])
        gammasum = np.zeros(self.nstate)
        gamma = np.zeros([ns, self.nstate])
        xisum = np.zeros([self.nstate, self.nstate])
        
        if self.ar1 == True:
            nodeprob_ar1hmm(self.nstate, self.covcube, 
                  self.coefcube, ymat, nodeprob)
            
            
        else:
            nodeprob_dtghmm(self.nstate, self.covcube, 
                  self.coefcube, ymat, xmat, nodeprob)
            
        filter_dtghmm(self.prior, self.tpm, nodeprob,
                gammasum, gamma, xisum)
        
        return gamma
    
    
    def predict_1step(self, ymat, xmat=None):
        '''
        one-step ahead prediction of observation and state probability
        
        Parameters
        ----------
        ymat: 2-d array
            observed univariate/multivariate times series
        xmat: 2-d array, default to None, only needed if self.ar1 == False.
            design matrix for exogenous variables. For 1st-order autoregressive hidden
            Markov model, xmat should be empty; for independent Gaussian hidden 
            Markov model, the first column of xmat should be 1 for intercept and the
            rest represents the covariates if any.
        '''
        ns = ymat.shape[0]
        nodeprob = np.zeros([ns, self.nstate])
        ncoly = ymat.shape[1]
        
        if self.ar1 == True:
            nodeprob_ar1hmm(self.nstate, self.covcube, 
                  self.coefcube, ymat, nodeprob)
            
        else:
            nodeprob_dtghmm(self.nstate, self.covcube, 
                  self.coefcube, ymat, xmat, nodeprob)
            
        
        alpha = np.zeros(self.nstate)
        alpha = self.prior * nodeprob[0,:]
        scale = np.dot(self.prior, nodeprob[0,:])
        alpha /= scale
        
        for i in range(1, ns):
            tempmat = np.dot(alpha, self.tpm)
            alpha = tempmat * nodeprob[i,:]
            scale = np.dot(tempmat, nodeprob[i,:])
            alpha /= scale
        
        probs = np.dot(alpha, self.tpm)
        thisy = ymat[ns-1,:]
        condobs = np.zeros([self.nstate, ncoly])
        obs = np.zeros(ncoly)
        
        if self.ar1 == True:
            for j in range(self.nstate):
                condobs[j,:] = self.coefcube[j,:,0] + np.dot(self.coefcube[j,:,1:],thisy)
            for k in range(ncoly):
                obs += condobs[j,:] * probs[j]
        else:
            thisx = np.mean(xmat, axis=0)
            for j in range(self.nstate):
                condobs[j,:] = np.dot(self.coefcube[j,:,:], thisx)
            for k in range(ncoly):
                obs += condobs[j,:] * probs[j]
            
        return obs, probs

    

###########################################################################3
#python function to simulate ar1-hmm
def sim_ar1hmm(n, prior, tpm, covcube, coefcube):
    
    nstate = tpm.shape[0]
    dim = covcube.shape[1]
 
    state = np.repeat(0, n) #integer
    y = np.repeat(0.0, n * dim).reshape(n, dim)

    state[0] = np.random.choice(nstate, 1, replace=False, p=prior)
    tempmu = coefcube[state[0],:,0]   
    tempcov = covcube[state[0],:,:]
    y[0,:] = multivariate_normal.rvs(mean=tempmu,cov=tempcov,size=1)
    #y[0,:] = np.random.multivariate_normal(tempmu, tempcov, 1) 
              
    for i in range(1, n):
        state[i] = np.random.choice(nstate, 1, replace=False,p=tpm[state[i-1],:])
        tempmu = coefcube[state[i],:,0]
        tempar = coefcube[state[i],:,1:]
        tempcov = covcube[state[i],:,:]
        tempmean = tempmu + np.dot(tempar, y[i-1, :])
        y[i,:] = multivariate_normal.rvs(mean=tempmean,cov=tempcov,size=1)
    return y, state 


#python function to simulate ghmm
def sim_ghmm(n, prior, tpm, covcube, coefcube, xmat):
    nstate = tpm.shape[0]
    dim = covcube.shape[1]
 
    state = np.repeat(0, n) #integer
    y = np.repeat(0.0, n * dim).reshape(n, dim)

    state[0] = np.random.choice(nstate, 1, replace=False, p=prior)
    tempbeta = coefcube[state[0],:,:]
    tempcov = covcube[state[0],:,:]
    tempmean = np.dot(tempbeta, xmat[0, :])
    y[0,:] = multivariate_normal.rvs(mean=tempmean,cov=tempcov,size=1)
              
    for i in range(1, n):
        state[i] = np.random.choice(nstate, 1, replace=False,p=tpm[state[i-1],:])
        tempbeta = coefcube[state[i],:,:]
        tempcov = covcube[state[i],:,:]
        tempmean = np.dot(tempbeta, xmat[i, :])
        y[i,:] = multivariate_normal.rvs(mean=tempmean,cov=tempcov,size=1)
    return y, state 



#cython function to compute node probability
@cython.boundscheck(False)
@cython.wraparound(False)
def nodeprob_ar1hmm(int nstate, double[:,:,:] covcube, 
                    double[:,:,:] coefcube,
                    double[:,:] y, double[:,:] result):
    cdef int y_nrow = y.shape[0]
    cdef int y_ncol = y.shape[1]
    cdef int i,j
    cdef double [:,:] chol = np.zeros([y_ncol,y_ncol])
    cdef double [:] vec1 = np.zeros([y_ncol])
    cdef double [:] vec2 = np.zeros([y_ncol])
    cdef double [:] vec00 = np.zeros([y_ncol])
    cdef double [:] vec01 = np.zeros([y_ncol])
    cdef double logpdf
    

    for i in range(y_nrow):
        for j in range(nstate):
            if i == 0:
                logpdf = 0
                cholesky(covcube[j,:,:], chol)
                minus(y[i,:], coefcube[j,:,0], vec1)
                solve_lower(chol, vec1, vec2)
                logpdf -= 0.5 * y_ncol * libc.math.log(2 * 3.1415926)
                logpdf -= sum_logdiag(chol)
                logpdf -= 0.5 * vec_mult(vec2,vec2)
                result[i,j] = libc.math.exp(logpdf)
            else:
                logpdf = 0
                cholesky(covcube[j,:,:], chol)
                matvec_mult(coefcube[j,:,1:], y[i-1,:], vec00)
                minus(y[i,:], coefcube[j,:,0], vec1)
                minus(vec1, vec00, vec01)
                solve_lower(chol, vec01, vec2)
                logpdf -= 0.5 * y_ncol * libc.math.log(2 * 3.1415926)
                logpdf -= sum_logdiag(chol)
                logpdf -= 0.5 * vec_mult(vec2,vec2)
                result[i,j] = libc.math.exp(logpdf)

#cython function to compute node probability
@cython.boundscheck(False)
@cython.wraparound(False)
def nodeprob_dtghmm(int nstate, double[:,:,:] covcube, 
                  double[:,:,:] coefcube,
                  double[:,:] y, double[:,:] xmat, double[:,:] result):
    cdef int y_nrow = y.shape[0]
    cdef int y_ncol = y.shape[1]
    cdef int i,j
    cdef double [:,:] chol = np.zeros([y_ncol,y_ncol])
    cdef double [:] vec1 = np.zeros([y_ncol])
    cdef double [:] vec2 = np.zeros([y_ncol])
    cdef double [:] vec00 = np.zeros([y_ncol])
    cdef double logpdf
    

    for i in range(y_nrow):
        for j in range(nstate):
            logpdf = 0
            cholesky(covcube[j,:,:], chol)
            matvec_mult(coefcube[j,:,:], xmat[i,:], vec00)
            minus(y[i,:], vec00, vec1)
            solve_lower(chol, vec1, vec2)
            logpdf -= 0.5 * y_ncol * libc.math.log(2 * 3.1415926)
            logpdf -= sum_logdiag(chol)
            logpdf -= 0.5 * vec_mult(vec2,vec2)
            result[i,j] = libc.math.exp(logpdf)

            

#cython forward backward algorithm
@cython.boundscheck(False)
@cython.wraparound(False)
def filter_dtghmm(double[:] prior, double[:,:] tpm, double[:,:] nodeprob,
                double[:] gammasum, double[:,:] gamma, double[:,:] xisum):
    
    cdef int nstate = tpm.shape[0]
    cdef int y_nrow = nodeprob.shape[0]
    cdef double loglik
    cdef double[:,:] tpm_t = np.zeros([nstate, nstate])
    cdef double[:] tempvec = np.zeros([nstate])
    cdef double[:,:] alpha = np.zeros([y_nrow, nstate])
    cdef double[:] scale = np.zeros([y_nrow])
    cdef double tempsum
    cdef double[:,:] tempmat = np.zeros([nstate,nstate])
    cdef double[:,:] tempmat2 = np.zeros([nstate,nstate])
    cdef int i,j,k
    cdef double[:,:] beta = np.zeros([y_nrow, nstate])
    cdef double[:,:] xi = np.zeros([y_nrow-1, nstate*nstate])
    #cdef double[:,:] xisum = np.zeros([nstate,nstate])
    
    mat_t(tpm, tpm_t)
    
    #start the forward part
    vec_mult_elem(prior, nodeprob[0,:], alpha[0,:])
    scale[0] = vec_mult(prior, nodeprob[0,:])
    for j in range(nstate):
        alpha[0,j] = alpha[0,j] / scale[0]
    
    loglik = libc.math.log(scale[0])
    
    #to use nogil, must manually type out all the functions
    #with cython.nogil:
    for i in range(1, y_nrow):
        matvec_mult(tpm_t, alpha[i-1,:],tempvec)
        vec_mult_elem(tempvec, nodeprob[i,:], alpha[i,:])
        scale[i] = vec_mult(tempvec, nodeprob[i,:])
        
        for j in range(nstate):
            alpha[i,j] = alpha[i,j] / scale[i]
        loglik += libc.math.log(scale[i])
    
    #start the backward part and posterior part
    #backward probs and state probs 
    for j in range(nstate):
        beta[y_nrow-1, j] = 1.0 / (nstate * scale[y_nrow-1])
        
    tempsum = vec_mult(alpha[y_nrow-1,:], beta[y_nrow-1,:])
    for j in range(nstate):
        gamma[y_nrow-1,j] = alpha[y_nrow-1,j] * beta[y_nrow-1, j] / tempsum
    
    for i in range(y_nrow-2, -1, -1):
        vec_mult_elem(beta[i+1,:], nodeprob[i+1,:], tempvec)
        matvec_mult(tpm, tempvec, beta[i,:])
        for j in range(nstate):
            beta[i,j] = beta[i,j] / scale[i]
        tempsum = vec_mult(alpha[i,:], beta[i,:])
        for j in range(nstate):
            gamma[i,j] = alpha[i,j] * beta[i, j] / tempsum

    
    for j in range(nstate):
        gammasum[j] = gamma[0,j]
        
    for i in range(1, y_nrow):
        for j in range(nstate):
            gammasum[j] += gamma[i,j]  
        
    #transition probs
    #xi = np.zeros([y_nrow-1, nstate**2])
    for i in range(y_nrow-1):
        vec_mult_elem(beta[i+1,:],nodeprob[i+1,:], tempvec)
        vec_outer(alpha[i,:], tempvec, tempmat)
        mat_mult_elem(tpm, tempmat, tempmat2)
        tempsum = 0
        for j in range(nstate):
            for k in range(nstate):
                xi[i, j+k*nstate] = tempmat2[j,k]
                tempsum += xi[i, j+k*nstate]
        for j in range(nstate):
            for k in range(nstate):
                xi[i, j+k*nstate] = xi[i, j+k*nstate] / tempsum
        
        for j in range(nstate):
            for k in range(nstate):
                xisum[j,k] = xi[0, j+k*nstate]
        
        for i in range(1, y_nrow-1):
            for j in range(nstate):
                for k in range(nstate):
                    xisum[j,k] += xi[i, j+k*nstate]

    return loglik


 
#cython E step summary for a single ghmm series
@cython.boundscheck(False)
@cython.wraparound(False)
def E_summary_dtghmm(int nstate, double [:] prior, double[:,:] tpm,
                     double[:,:,:] covcube, 
                     double[:,:,:] coefcube, double[:,:] ymat,
                     double[:,:] xmat, 
                     double[:,:] gamma,double coefshrink,
                     double[:] gammasum, double[:,:] xisum, 
                     double[:,:,:] xwx, double[:,:,:] xwy):
    cdef int ynrow = ymat.shape[0]
    cdef int yncol = ymat.shape[1]
    cdef int xncol = xmat.shape[1]
    cdef int i,j
    cdef double nllk
    
    cdef double[:,:] nodeprob = np.zeros([ynrow, nstate])
    #cdef double[:,:] gamma = np.zeros([ynrow, nstate])
    
    nodeprob_dtghmm(nstate, covcube, 
                  coefcube, ymat, xmat, nodeprob)
    
    nllk = -filter_dtghmm(prior, tpm, nodeprob,
                       gammasum, gamma, xisum)
    
  
    for i in range(nstate):
        mult_XWX(xmat, gamma[:,i], coefshrink, xwx[i,:,:])
        for j in range(yncol):
            mult_XWy(xmat, gamma[:,i], ymat[:,j], xwy[i,:,j])
            
    return nllk

#cython E step summary for a single ghmm series
@cython.boundscheck(False)
@cython.wraparound(False)
def E_summary_ar1hmm(int nstate, double [:] prior, double[:,:] tpm,
                     double[:,:,:] covcube, 
                     double[:,:,:] coefcube, double[:,:] ymat,
                     double coefshrink, double[:,:] gamma,
                     double[:] gammasum, double[:,:] xisum, 
                     double[:,:,:] xwx, double[:,:,:] xwy):
    cdef int ynrow = ymat.shape[0]
    cdef int yncol = ymat.shape[1]
 
    cdef int i,j
    cdef double nllk
    
    cdef double[:,:] nodeprob = np.zeros([ynrow, nstate])
    #cdef double[:,:] gamma = np.zeros([ynrow, nstate])
    cdef double[:,:] newx = np.ones([ynrow-1, yncol+1])
    cdef double[:,:] newy = np.ones([ynrow-1, yncol])

    nodeprob_ar1hmm(nstate, covcube, 
                  coefcube, ymat, nodeprob)
    
    nllk = -filter_dtghmm(prior, tpm, nodeprob,
                       gammasum, gamma, xisum)
    
    for i in range(ynrow-1):
        for j in range(yncol):
            newx[i, j+1] = ymat[i,j]
            newy[i, j] = ymat[i+1, j]
            
    for i in range(nstate):
        mult_XWX(newx, gamma[1:,i], coefshrink, xwx[i,:,:])
        for j in range(yncol):
            mult_XWy(newx, gamma[1:,i], newy[:,j], xwy[i,:,j])
            
    return nllk

@cython.boundscheck(False)
@cython.wraparound(False)
def EM_ar1hmm(int nstate, double[:]prior, double[:,:]tpm, double[:,:,:]covcube, 
             double[:,:,:]coefcube, long[:] ntimes,
             double[:,:] ymat, 
             double coefshrink,double covshrink):
    
    cdef int yncol = ymat.shape[1]
    cdef int nsubj = len(ntimes)
    cdef double[:] gammasum = np.zeros(nstate)
    cdef double[:,:] xisum = np.zeros([nstate,nstate])
    cdef double[:,:,:] xwx = np.zeros([nstate,yncol+1,yncol+1])
    cdef double[:,:,:] xwy = np.zeros([nstate,yncol+1,yncol])
    cdef double nllk = 0.0
    cdef int i,j,k,l
    cdef double rowsum = 0.0
    
    cdef double[:] tgammasum = np.zeros(nstate)
    cdef double[:,:] txisum = np.zeros([nstate,nstate])
    cdef double[:,:,:]txwx = np.zeros([nstate,yncol+1,yncol+1])
    cdef double[:,:,:]txwy = np.zeros([nstate,yncol+1,yncol])
    
    cdef long n = vecsum(ntimes)
    cdef double[:,:] gamma = np.zeros([n, nstate])
    cdef double[:,:] newgamma = np.zeros([n - nsubj, nstate])
    cdef double[:,:] newx = np.ones([n - nsubj, yncol+1])
    cdef double[:,:] newy = np.ones([n - nsubj, yncol])

    #E step
    for i in range(nsubj):
        
        if i == 0:
            starting = 0
            ending = ntimes[0]
            newstart = 0
            newend = ntimes[0] - 1
        else:
            starting = ending
            ending = ending + ntimes[i]
            newstart = newend
            newend = newend + ntimes[i] - 1
            
        nllk += E_summary_ar1hmm(nstate, prior, tpm,
                       covcube, coefcube, ymat[starting:ending,:], 
                       coefshrink, gamma[starting:ending, :],
                       tgammasum, txisum, txwx, txwy)
        
        
        for j in range(newend - newstart):
            for k in range(yncol):
                newx[newstart+j, k+1] = ymat[starting+j,k]
                newy[newstart+j, k] = ymat[starting+j+1,k]
            for k in range(nstate):
                newgamma[newstart+j,k] = gamma[starting+j+1,k]
                
        
        for j in range(nstate):
            gammasum[j] += tgammasum[j]
        
        for j in range(nstate):
            for k in range(nstate):
                xisum[j,k] += txisum[j,k]
        
        for j in range(nstate):
            for k in range(yncol+1):
                for l in range(yncol+1):
                    xwx[j,k,l] += txwx[j,k,l]
                    
        for j in range(nstate):
            for k in range(yncol+1):
                for l in range(yncol):
                    xwy[j,k,l] += txwy[j,k,l]
                    
    #M step
    for j in range(nstate):
        prior[j] = gammasum[j] / n
    
    for i in range(nstate):
        rowsum = 0.0
        for j in range(nstate):
            tpm[i,j] = xisum[i,j]
            rowsum += xisum[i,j]
        for j in range(nstate):
            tpm[i,j] = tpm[i,j] / rowsum
 
    
    coef_cov(nstate, xwx, xwy, newx, newy, newgamma, gammasum,
             covshrink, coefcube, covcube)
    
    return nllk

@cython.boundscheck(False)
@cython.wraparound(False)
def EM_dtghmm(int nstate, double[:]prior, double[:,:]tpm, double[:,:,:]covcube, 
             double[:,:,:]coefcube, long[:] ntimes,
             double[:,:] xmat, double[:,:] ymat, 
             double coefshrink,double covshrink):
    
    cdef int xncol = xmat.shape[1]
    cdef int yncol = ymat.shape[1]
    cdef int nsubj = len(ntimes)
    cdef double[:] gammasum = np.zeros(nstate)
    cdef double[:,:] xisum = np.zeros([nstate,nstate])
    cdef double[:,:,:] xwx = np.zeros([nstate,xncol,xncol])
    cdef double[:,:,:] xwy = np.zeros([nstate,xncol,yncol])
    cdef double nllk = 0.0
    cdef int i,j,k,l
    cdef double rowsum = 0.0
    
    cdef long n = vecsum(ntimes)
    cdef double[:,:] gamma = np.zeros([n, nstate])
    cdef double[:] tgammasum = np.zeros(nstate)
    cdef double[:,:] txisum = np.zeros([nstate,nstate])
    cdef double[:,:,:]txwx = np.zeros([nstate,xncol,xncol])
    cdef double[:,:,:]txwy = np.zeros([nstate,xncol,yncol])
    
    #E step
    for i in range(nsubj):
        
        if i == 0:
            starting = 0
            ending = ntimes[0]
        else:
            starting = ending
            ending = ending + ntimes[i]
            
        nllk += E_summary_dtghmm(nstate, prior, tpm,
                       covcube, coefcube, ymat[starting:ending,:],
                       xmat[starting:ending,:], gamma[starting:ending, :],
                       coefshrink, 
                       tgammasum, txisum, txwx, txwy)
        
        for j in range(nstate):
            gammasum[j] += tgammasum[j]
        
        for j in range(nstate):
            for k in range(nstate):
                xisum[j,k] += txisum[j,k]
        
        for j in range(nstate):
            for k in range(xncol):
                for l in range(xncol):
                    xwx[j,k,l] += txwx[j,k,l]
                    
        for j in range(nstate):
            for k in range(xncol):
                for l in range(yncol):
                    xwy[j,k,l] += txwy[j,k,l]
                    
    #M step
    for j in range(nstate):
        prior[j] = gammasum[j] / n
    
    for i in range(nstate):
        rowsum = 0.0
        for j in range(nstate):
            tpm[i,j] = xisum[i,j]
            rowsum += xisum[i,j]
        for j in range(nstate):
            tpm[i,j] = tpm[i,j] / rowsum
 
    
    coef_cov(nstate, xwx, xwy, xmat, ymat, gamma, gammasum,
             covshrink, coefcube, covcube)
    
    return nllk


@cython.boundscheck(False)
@cython.wraparound(False)
def coef_cov(int nstate, double[:,:,:] xwx, double[:,:,:] xwy, double[:,:] xmat,
             double[:,:] ymat, double[:,:] gamma, double[:] gammasum, double covshrink,
             double[:,:,:] coefcube, double[:,:,:] covcube):
    
    cdef int m = xmat.shape[0]
    cdef int n = xmat.shape[1]
    cdef int p = ymat.shape[1]
    cdef double[:] yhat = np.zeros([m])
    cdef double[:,:] emat = np.zeros([m,p])
    cdef double[:,:] chol = np.zeros([n,n])
    cdef double[:] tempvec = np.zeros(n)
    
    cdef double[:] tempvec2 = np.zeros(m)
    cdef double[:,:] tchol = np.zeros([n,n])
    cdef double tr
    
    cdef int i,j,k
    for i in range(nstate):
        cholesky(xwx[i,:,:], chol)
        mat_t(chol, tchol)
        
        for j in range(p):
            solve_lower(chol, xwy[i,:,j], tempvec)
            solve_upper(tchol, tempvec, coefcube[i,j,:])
            matvec_mult(xmat, coefcube[i,j,:], yhat)
            minus(ymat[:,j], yhat, emat[:, j])
            
        for j in range(p):
            vec_mult_elem(emat[:,j], gamma[:,i], tempvec2)
            for k in range(p):
                
                covcube[i,j,k] = vec_mult(tempvec2, emat[:,k]) / gammasum[i]
        
        tr = avgtrace(covcube[i,:,:])
        for j in range(p):
            for k in range(p):
                if j == k:
                    covcube[i,j,k] = (1-covshrink)*covcube[i,j,k] + covshrink*tr
 

  
###################################################################
'''
utility functions
'''    
########################################################
#vector substraction              
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void minus(double[:] vec1, double[:] vec2, double[:] vec3):
    cdef int dim = len(vec1)
    cdef int i
    with cython.nogil:
        for i in range(dim):
            vec3[i] = vec1[i] - vec2[i]

#matrix transpose
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void mat_t(double[:,:] mat, double[:,:] output):
    
    cdef int nrow = mat.shape[0]
    cdef int ncol = mat.shape[1]
    cdef int i,j
    cdef double temp
    
    with cython.nogil:
        for i in range(nrow):
            for j in range(ncol):
                if i != j:
                    output[i,j] = mat[j,i]
                else:
                    output[i,j] = mat[i,j]
#matrix columnsum
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void colsum(double[:,:] mat, double[:] columnsum):
    cdef int row = mat.shape[0]
    cdef int col = mat.shape[1]
    cdef int i,j
    
    with cython.nogil:
        for j in range(col):
            columnsum[j] = 0
        for i in range(row):
            for j in range(col):
                columnsum[j] += mat[i,j]
           
#vectorize ---- axis = 0 columnwise, 1 rowwise
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void vect(double[:,:] mat, int axis, double[:] result):
    cdef int row = mat.shape[0]
    cdef int col = mat.shape[1]
    cdef int i,j
    with cython.nogil:
        for i in range(row):
            for j in range(col):
                if axis == 0:  #by column
                    result[i + j*row] = mat[i,j]
                else:  #by row
                    result[j + i*col] = mat[i,j]
            

    
#function for matrix multiplication
#everything like C (res is a pointer)  
@cython.boundscheck(False)
@cython.wraparound(False)     
cdef void matrix_multiply(double[:,:] u, double[:, :] v, double[:, :] res):
    cdef int i, j, k
    cdef int m = u.shape[0];
    cdef int n = u.shape[1];
    cdef int p = v.shape[1];

    with cython.nogil:
        for i in range(m):
            for j in range(p):
                res[i,j] = 0
                for k in range(n):
                    res[i,j] += u[i,k] * v[k,j]

#function for X' W X where W is a diagonal matrix
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void mult_XWX(double[:,:]X, double [:] w, double shrink,
                   double [:,:] res):
    cdef int i,j,k
    cdef int m = X.shape[0]
    cdef int n = X.shape[1]
    cdef double tempsum;
    
    with cython.nogil:
        for i in range(n):
            for j in range(n):
                if j < i:
                    res[i,j] = res[j,i]
                elif j == i:
                    res[i,j] = shrink
                    for k in range(m):
                        res[i,j] += w[k] * X[k,i] * X[k,j]
                else:
                    res[i,j] = 0
                    for k in range(m):
                        res[i,j] += w[k] * X[k,i] * X[k,j]


#function for X' W Y where W is a diagonal matrix
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void mult_XWy(double[:,:] X, double [:] w, 
                   double[:] y, double [:] res):
    cdef int i,k
    cdef int m = X.shape[0]
    cdef int n = X.shape[1]
    cdef double tempsum;
    
    with cython.nogil:
        for i in range(n):
            res[i] = 0
            for k in range(m):
                res[i] += w[k] * X[k,i] * y[k]
                

#function for matrix multiplication
#everything like C (res is a pointer)  
@cython.boundscheck(False)
@cython.wraparound(False)     
cdef void mat_mult_elem(double[:,:] u, double[:, :] v, double[:, :] res):
    cdef int i, j
    cdef int m = u.shape[0];
    cdef int n = u.shape[1];
    
    with cython.nogil:
        for i in range(m):
            for j in range(n):
                res[i,j] = u[i,j] * v[i,j]

#matrix multiplied by vector
@cython.boundscheck(False)
@cython.wraparound(False)     
cdef void matvec_mult(double[:,:] u, double[:] v, double[:] res):
    cdef int i, j, k
    cdef int m = u.shape[0];
    cdef int n = u.shape[1];

    with cython.nogil:
        for i in range(m):
            res[i] = 0
            for k in range(n):
                res[i] += u[i,k] * v[k]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void vec_outer(double[:] vec1, double[:] vec2, double[:,:] result):
    cdef int row = len(vec1)
    cdef int col = len(vec2)
    
    with cython.nogil:
        for i in range(row):
            for j in range(col):
                result[i,j] = vec1[i] * vec2[j]
            
    
    
#VECTOR dotproduct
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double vec_mult(double[:] vec1, double[:] vec2):
    cdef int dim = len(vec1)
    cdef double result = 0
    with cython.nogil:
        for i in range(dim):
            result += vec1[i] * vec2[i]
    return result

#elementwise multiplication between vectors
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void vec_mult_elem(double[:] vec1, double[:] vec2, double[:] vec3):
    cdef int dim = len(vec1)
    with cython.nogil:
        for i in range(dim):
            vec3[i] = vec1[i] * vec2[i]


#function for sum of log diagonals
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double sum_logdiag(double[:,:] mat):
    cdef double res = 0
    cdef double temp
    cdef int dim = mat.shape[0]
    cdef int i
    
    with cython.nogil:
        for i in range(dim):
            temp = libc.math.log(mat[i,i])
            res += temp

    return res
                  
#function for solving lower triangular array
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void solve_lower(double[:,:] A, double[:] b, double[:] x):
    
    cdef int i,j
    cdef double s
    cdef int n = A.shape[0]
    
    with cython.nogil:
        for i in range(n):
            s = 0
            for j in range(i):
                s += A[i,j] * x[j]
            
            x[i] = (b[i] - s) / A[i,i]
  
#function for solving upper triangular array
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void solve_upper(double[:,:] A, double[:] b, double[:] x):
    
    cdef int i,j
    cdef double s
    cdef int n = A.shape[0]
    
    with cython.nogil:
        
        for i in range(n):
            x[i] = 0
            
        for i in range(n-1, -1, -1):
            s = 0
            for j in range(n-1, i-1, -1):
                s += A[i,j] * x[j]
            
            x[i] = (b[i] - s) / A[i,i]
                    
                    
#function for cholesky decomposition
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cholesky(double[:, :] p, double[:, :] L):
    cdef int i,j,k
    cdef double temp1, temp2
    cdef int dim = p.shape[0]
    
    with cython.nogil:
        for i in range(dim):
            for j in range(dim):
                temp1 = 0
                temp2 = 0
                if i > j:
                    if j > 0:
                        for k in range(1, j+1):
                            temp2 += L[i,k-1] * L[j,k-1]
                    
                    L[i,j] = (p[i,j] - temp2) / L[j,j]
                
                elif i == j:
                    for k in range(i):
                        temp1 += L[i,k] * L[i,k]
                    L[i,j] = libc.math.sqrt(p[i,j] - temp1)
                    
                else:
                    L[i,j] = 0
                    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef long vecsum(long[:]vec):
    cdef int dim = len(vec)
    cdef int i
    cdef long result = 0
    for i in range(dim):
        result += vec[i]
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double avgtrace(double[:,:]mat):
    cdef int dim = mat.shape[0]
    cdef double tempsum = 0
    cdef int i
    with cython.nogil:
        for i in range(dim):
            tempsum += mat[i,i]
    return tempsum/dim



#function to convert hsmm to hmm regarding tpm
#each row in dm is a dwell time pmf, mv is a vector of truncations
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void hsmm_hmm(double[:,:] omega, double[:,:] dm, int[:] mv, double[:,:] gamma):
    cdef int m = omega.shape[0]
    cdef dmrow = dm.shape[0]
    cdef dmcol = dm.shape[1] #maximum truncation across states
    cdef int dim = 0;
    cdef int i,j,p,q,mi,rowsum,colsum
    
    cdef double[:,:] temp = np.zeros([dmrow,dmcol])
    cdef double[:,:] ci = np.zeros([dmrow,dmcol])
    cdef double[:,:] cim = np.zeros([dmrow,dmcol])
    
    #this is the dimension of the final result(i.e. gamma)
    for i in range(m):
        dim += mv[i]
    
    with cython.nogil:
        for i in range(m):
            mi = mv[i]
            for j in range(mi):
                if j == 0:
                    temp[i,j] = 0
                else:
                    temp[i,j] = temp[i,j-1] + dm[i,j-1]
            
            for j in range(mi):
                if libc.math.fabs(1-temp[i,j]) > 0.0000001:
                    ci[i,j] = dm[i,j] / (1-temp[i,j])
                else:
                    ci[i,j] = 1
                if 1 - ci[i,j] > 0:
                    cim[i,j] = 1 - ci[i,j]
                else:
                    cim[i,j] = 0
        
        rowsum = 0
        
        for i in range(m):
            colsum = 0
            for j in range(m):
                if i == j:
                    if mv[i] == 1:
                        gamma[rowsum,colsum] = cim[i,0]
                    else:
                        for p in range(mv[i]):
                            for q in range(mv[j]):
                                if q - p == 1:
                                    gamma[rowsum+p, colsum+q] = cim[i,p]
                                elif p == mv[i] - 1 and q == mv[j] - 1:
                                    gamma[rowsum+p, colsum+q] = cim[i,p]
                                else:
                                    gamma[rowsum+p, colsum+q] = 0
                else:
                    for p in range(mv[i]):
                        for q in range(mv[j]):
                            if q == 0:
                                gamma[rowsum+p, colsum+q] = omega[i,j] * ci[i,p]
                            else:
                                gamma[rowsum+p, colsum+q] = 0
                                
                colsum += mv[j]
            rowsum += mv[i]
    
def tryhsmmhmm(omega,dm,mv):
    dim = np.sum(mv)
    gamma = np.zeros([dim,dim])
    hsmm_hmm(omega,dm,mv,gamma)
    return gamma

#python function to convert hsmm to hmm
def hsmm_hmm_py(omega, dm, mv):
    m = omega.shape[0]
    dmrow = dm.shape[0]
    dmcol = dm.shape[1]
    dim = np.sum(mv)
    
    temp = np.zeros(dm.shape)
    ci = np.zeros(dm.shape)
    cim = np.zeros(dm.shape)
    gamma = np.zeros([dim,dim])
    
    for i in range(m):
        mi = mv[i]
        for j in range(mi):
            if j==0:
                temp[i,j] = 0
            else:
                temp[i,j] = temp[i,j-1] + dm[i,j-1]
        for j in range(mi):
            if np.abs(1-temp[i,j]) > 0.0000001:
                ci[i,j] = dm[i,j] / (1-temp[i,j])
            else:
                ci[i,j] = 1
            if 1 - ci[i,j] > 0:
                cim[i,j] = 1 - ci[i,j]
            else:
                cim[i,j] = 0
    rowsum = 0
    for i in range(m):
        colsum = 0
        for j in range(m):
            if i==j:
                if mv[i]==1:
                    gamma[rowsum, colsum] = cim[i,0]
                else:
                    for p in range(mv[i]):
                        for q in range(mv[j]):
                            if q-p==1:
                                gamma[rowsum+p, colsum+q] = cim[i,p]
                            elif p==mv[i]-1 and q==mv[j]-1:
                                gamma[rowsum+p, colsum+q] = cim[i,p]
                            else:
                                gamma[rowsum+p, colsum+q] = 0
                
                
            else:
                for p in range(mv[i]):
                    for q in range(mv[j]):
                        if q==0:
                            gamma[rowsum+p, colsum+q] = omega[i,j] * ci[i,p]
                        else:
                            gamma[rowsum+p, colsum+q] = 0
            colsum += mv[j]
        rowsum += mv[i]
    return gamma