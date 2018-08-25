import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy import identity
#from scipy.sparse import identity 
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve
 

class dlm:
    
    """Dynamic linear model
    
    y_t = A_t * x_t + w_t   (R)
    x_t = PHI * x_t-1 + v_t  (Q)
    x_0 ~ N(mu0, SIGMA0)
    
    Parameters
    ----------

    mu0: 1-day array.
        prior mean for the initial latent variable.
    SIGMA0: 2-d array.
        prior covariance for the initial latent variable   
    A: 3-d array
       The time-varying coefficient matrix at each observed time. The first index is the 
       number of observations. The second index is the dimension of observation.
       The third index is the dimension of latent variables.
    R: 2-d array
       covariance matrix for the observations.
    PHI: 2-d array
       transition matrix for the latent variables.
    Q: 2-d array
       covariance matrix for the latent variables.
    """
 
    def __init__(self, mu0, SIGMA0, A, R, PHI, Q):
        self.mu0 = mu0
        self.SIGMA0 = SIGMA0
        self.A = A
        self.R = R
        self.PHI = PHI
        self.Q = Q


    def simulate(self, ns):
        """
        simulate the observations and latent variables for dynamic linear model
        
        Parameters
        ----------
        
        ns: int
            length of the times series
        """
        y_nrow = ns
        x_nrow = ns
        y_ncol = self.A.shape[1]
        x_ncol = self.A.shape[2]
        y = np.zeros([y_nrow,y_ncol])
        x = np.zeros([x_nrow,x_ncol])
        x[0,:] = multivariate_normal.rvs(mean=self.mu0, cov=self.SIGMA0, size=1) 
        y[0,:] = multivariate_normal.rvs(mean=np.dot(self.A[0,:,:],x[0,:]),\
                     cov=self.R,size=1)
        
        for t in range(1, y_nrow):
            x[t,:] = multivariate_normal.rvs(mean=np.dot(self.PHI,x[t-1,:]),\
                    cov=self.Q,size=1)
            y[t,:] = multivariate_normal.rvs(mean=np.dot(self.A[t,:,:],x[t,:]),\
                    cov=self.R,size=1)
        
        return x, y

    def filtering(self, y):
        """
        Kalman filtering step
        
        Parameters
        ----------
        
        y: 2-d array
            observed time series
        """
        y_nrow = y.shape[0]
        y_ncol = y.shape[1]
        x_nrow = y.shape[0]
        x_ncol = self.A.shape[2]
            
        x_filter = np.zeros([x_nrow,x_ncol])
        x_pred = np.zeros([x_nrow,x_ncol])
        P_filter = np.zeros([y_nrow, x_ncol, x_ncol])
        P_pred = np.zeros([y_nrow, x_ncol, x_ncol])
            
        #initialize
        #substitute missing values
        newy, newA, newR, R22 = miss(y[0,:],self.A[0,:,:],self.R)
            
        x_pred[0,:] = np.dot(self.PHI, self.mu0)
        P_pred[0,:,:] = np.dot(self.PHI, np.dot(self.SIGMA0,self.PHI.T)) + self.Q
        temp = np.dot(np.dot(newA,P_pred[0,:,:]), newA.T) + newR
        K = solve(temp.T, np.dot(newA, P_pred[0,:,:].T)).T
        x_filter[0,:] = x_pred[0,:] + np.dot(K, newy-np.dot(newA,x_pred[0,:]))
        P_filter[0,:,:] = np.dot(identity(x_ncol) - np.dot(K,newA), P_pred[0,:,:])

        for t in range(1,y_nrow):
            newy, newA, newR,R22 = miss(y[t,:],self.A[t,:,:],self.R)
                
            x_pred[t,:] = self.PHI.dot(x_filter[t-1,:]) 
            P_pred[t,:,:] = np.dot(np.dot(self.PHI, P_filter[t-1,:,:]), self.PHI.T) + self.Q
                
            temp = np.dot(np.dot(newA,P_pred[t,:,:]), newA.T) + newR
            K = solve(temp.T, np.dot(newA, P_pred[t,:,:].T)).T
            x_filter[t,:] = x_pred[t,:] + np.dot(K, newy-np.dot(newA,x_pred[t,:]))
            P_filter[t,:,:] = np.dot(identity(x_ncol) - np.dot(K,newA), P_pred[t,:,:])

        return x_pred,x_filter,P_pred,P_filter, K
    
    
    def smoothing(self, y):
        """
        Kalman smoothing step
        
        Parameters
        ----------
        
        y: 2-d array
            observed time series
        """
        y_nrow = y.shape[0]
        y_ncol = y.shape[1]
        x_nrow = y.shape[0]
        x_ncol = self.A.shape[2]
        x_pred,x_filter,P_pred,P_filter,K = self.filtering(y)
            
        J = np.zeros([y_nrow-1, x_ncol, x_ncol])
        x_smooth = np.zeros([x_nrow,x_ncol])
        P_smooth = np.zeros([y_nrow, x_ncol, x_ncol])
        P_cov_smooth = np.zeros([y_nrow-1, x_ncol, x_ncol])
            
        #initialize
        x_smooth[y_nrow-1,:] = x_filter[y_nrow-1,:]
        P_smooth[y_nrow-1,:,:] = P_filter[y_nrow-1,:,:]
        
        newy, newA, newR, R22 = miss(y[y_nrow-1,:],self.A[y_nrow-1,:,:],self.R)
        P_cov_smooth[y_nrow-2,:,:] = np.dot(identity(x_ncol)-np.dot(K,newA),\
                    np.dot(self.PHI, P_filter[y_nrow-2,:,:]))
        
        #induction
        for t in range(y_nrow-1,0,-1):
            temp = np.dot(P_filter[t-1,:,:], self.PHI.T)
            J[t-1,:,:] = solve(P_pred[t,:,:].T, temp.T).T
            x_smooth[t-1,:] = x_filter[t-1,:] + np.dot(J[t-1,:,:], x_smooth[t,:]-x_pred[t,:])
            P_smooth[t-1,:,:] = P_filter[t-1,:,:]+np.dot(J[t-1,:,:],\
                    np.dot(P_smooth[t,:,:]-P_pred[t,:,:], J[t-1,:,:].T))
        
        #covariance smoother
        for t in range(y_nrow-1,1,-1):
            P_cov_smooth[t-2,:,:] = np.dot(P_filter[t-1,:,:],J[t-2,:,:].T)+\
                np.dot(J[t-1,:,:], np.dot(P_cov_smooth[t-1,:,:]-np.dot(self.PHI,P_filter[t-1,:,:]),\
                   J[t-2,:,:].T ))
       
        return x_smooth, P_smooth, P_cov_smooth, x_pred, P_pred
    
    
    def fb(self,y):
        """
        Forward filtering backward smoothing
        
        Parameters
        ----------
        
        y: 2-d array
            observed time series
        """
        y_nrow = y.shape[0]
        y_ncol = y.shape[1]
        x_nrow = y.shape[0]
        x_ncol = self.A.shape[2]
        
        loglik = 0
        xs,ps,pcov,x_pred,P_pred = self.smoothing(y)
        
     
        S11 = np.outer(xs[0,:], xs[0,:]) + ps[0,:,:]
        S10 = np.zeros(self.Q.shape)
        S00 = np.zeros(self.Q.shape)
            
        newy, newA, newR,R22 = miss(y[0,:],self.A[0,:,:],self.R)
            
        rsum = np.outer(newy-np.dot(newA,xs[0,:]), newy.T-np.dot(newA,xs[0,:]).T)+\
                   np.dot(newA,np.dot(ps[0,:,:],newA.T))
            
        sigma = np.dot(newA, np.dot(P_pred[0,:,:],newA.T)) + newR
        epsilon = newy - np.dot(newA, x_pred[0,:])
        #this is the observed data likelihood
        loglik = -0.5*(np.log(np.trace(sigma)) + np.dot(epsilon.T,\
                     np.dot(sigma, epsilon)))
                
        for t in range(1, y_nrow):
                
            S11 += np.outer(xs[t,:], xs[t,:]) + ps[t,:,:]
            S10 += np.outer(xs[t,:], xs[t-1,:]) + pcov[t-1,:,:]
            S00 += np.outer(xs[t-1,:], xs[t-1,:]) + ps[t-1,:,:]
                
            newy, newA, newR, R22 = miss(y[t,:],self.A[t,:,:],self.R)
                
            rsum += np.outer(newy-np.dot(newA,xs[t,:]), newy.T-np.dot(newA,xs[t,:]).T)+\
                     np.dot(newA,np.dot(ps[t,:,:],newA.T))
            sigma = np.dot(newA, np.dot(P_pred[t,:,:],newA.T)) + self.R
            epsilon = newy - np.dot(newA, x_pred[t,:])
            loglik -= 0.5*(np.log(np.trace(sigma)) + np.dot(epsilon.T,\
                          np.dot(sigma, epsilon))) 
            
        return S00,S10,S11,rsum,xs[0,:],ps[0,:,:],loglik
    
     
    def EM(self,y, ntimes, estPHI=True, maxit=100, tol=1e-4,verbose=True):
        """
        offline EM algorithm
        
        Parameters
        ----------
        
        y: 2-d array
            observed time series
        ntimes: 1-d array
            length of each individual series
        estPHI: boolean, default to True
            if False, PHI (transition matrix for latent variables) is set
            to be identity matrix rather than estimating it
        maxit: int, default to 100
            maximum number of EM iteration
        tol: int, default to 1e-4
            tolerance for the EM algorithm to stop
        verbose: boolean, default to True
            whether to print the information for each iteration
        """
        y_nrow = y.shape[0]
        y_ncol = y.shape[1]
        x_nrow = y.shape[0]
        x_ncol = self.A.shape[2]
        nsubj = len(ntimes)
        cumsum = np.cumsum(ntimes)
        
        bigA = self.A
        
        iteration = 0
        
                
        while iteration < maxit:
            iteration += 1
                
            for n in range(nsubj):
                if n==0:
                    index = range(0,cumsum[n])
                    ysub = y[index,:]
                    self.A = bigA[index,:,:]
                    S00,S10,S11,rsum,xs,ps,loglik = self.fb(ysub)
                        
                else:
                    index = range(cumsum[n-1],cumsum[n])
                    ysub = y[index,:]
                    self.A = bigA[index,:,:]
                    tS00,tS10,tS11,trsum,txs,tps,tloglik = self.fb(ysub)
                        
                    S00 += tS00
                    S10 += tS10
                    S11 += tS11
                    rsum += trsum
                    xs += txs
                    ps += tps
                    loglik += tloglik
                         
            if iteration==1: likbase = loglik
            if iteration>2:
                if np.abs(loglik - oldlik)/np.abs(oldlik) < tol or loglik < oldlik:
                    loglik = oldlik
                    break
                
            if estPHI == True:
                PHI = solve(S00.T, S10.T).T
            else:
                PHI = self.PHI
                
            Q = (S11+np.dot(PHI,np.dot(S00,PHI.T))-np.dot(PHI,S10.T)-\
                         np.dot(S10,PHI.T))/y_nrow

                    
            R = rsum / y_nrow
            mu0 = xs / nsubj
            SIGMA0 = ps / nsubj
            oldlik = loglik
            if verbose == True and iteration>1:
                print("iteration = ", iteration, ";negloglik = ", -loglik)
                
            self.A = bigA
            self.PHI = PHI
            self.mu0 = mu0
            self.SIGMA0 = SIGMA0
            self.Q = Q
            self.R = R
            
        return -loglik
            

    
    def onestep_forecast(self, x_filter, P_filter):
        """
        one step ahead forecasting
        
        Parameters
        ----------
        
        x_filter: 1-d array
            filtered latent variable at the current time
        P_filter: 2-d array
            filtered covariance matrix for latent variable at the current time
        """
        x_forecast = self.PHI.dot(x_filter)
        P_forecast = np.dot(np.dot(self.PHI, P_filter), self.PHI.T) + self.Q
        
        return x_forecast, P_forecast
    
     
    def onestep_filter(self, y_curr, x_pred, P_pred, A_curr):
        """
        one step ahead filtering
        
        Parameters
        ----------
        y_curr: 1-d array
            current observation
        x_pred: 1-d array
            forecasted latent variable at the current time
        P_pred: 2-d array
            forecasted covariance matrix for latent variable at the current time
        A_curr: 2-d array
            current coefficient matrix
        """
        
        x_ncol = A_curr.shape[1]
        
        newy, newA, newR,R22 = miss(y_curr,A_curr,self.R)
        temp = np.dot(np.dot(newA,P_pred), newA.T) + newR
        K = solve(temp.T, np.dot(newA, P_pred.T)).T
        x_filter = x_pred + np.dot(K, newy-np.dot(newA,x_pred))
        P_filter = np.dot(identity(x_ncol) - np.dot(K,newA), P_pred)
            
         
        return x_filter, P_filter   
    
####################################################################################
#substitute missing values
def miss(yrow, Acurr, Rcurr):
    
    rdim = Rcurr.shape[0]
    
    newy = yrow.copy()
    newA = Acurr.copy()
    newR = Rcurr.copy()
    R22 = np.zeros(Rcurr.shape)
    check = pd.isnull(yrow)
    indices = [k for k, val in enumerate(check) if check[k]]  
    newy[indices] = 0.0
    newA[indices,:] = 0.0
    for i in indices:
        for j in range(rdim):
            if i == j:
                newR[i,j] = 1.0
                R22[i,j] = 1.0
            else:
                newR[i,j] = 0.0
    return newy, newA, newR, R22
    
    