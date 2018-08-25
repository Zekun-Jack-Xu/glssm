import os
import numpy as np
 
import matplotlib.pyplot as plt

from glssm import *
#################################################
#A constant, dim y = 1, dim x = 1, randomwalk(estPHI=False)
xdict = dict()
ydict = dict()
nsubj = 20
ns = np.zeros(nsubj, dtype=np.int)


mu0 = np.array([0.0])
SIGMA0 = np.array([1.0]).reshape(1,1)
R = np.array([2.5]).reshape(1,1)
PHI = np.array([1.0]).reshape(1,1)
Q = np.array([1.5]).reshape(1,1)


for i in range(nsubj):
    ns[i] = int(np.random.uniform(200,400,1))
    A = np.ones([ns[i],1,1])
    s1 = dlm(mu0,SIGMA0,A,R,PHI,Q)
    tempx, tempy = s1.simulate(ns[i])
    ydict.update({i: tempy})
    xdict.update({i: tempx})

bigxmat = xdict[0]
bigns = [ns[0]]
bigymat = ydict[0]

for i in range(1,nsubj):
    bigxmat = np.vstack((bigxmat, xdict[i]))
    bigymat = np.vstack((bigymat, ydict[i]))
    bigns.append(ns[i])
    
    
ntimes = np.array(bigns) 

#check the first series
tempy = ydict[0]
tempx = xdict[0]
tempy[50:70] = None
A = np.ones([ns[0],1,1])
s1 = dlm(mu0,SIGMA0,A,R,PHI,Q)
xp,xf,pp,pf,K_last = s1.filtering(tempy)
xs,ps,pcov,xp,pp = s1.smoothing(tempy)

#plots
fig, ax = plt.subplots()
plt.plot(tempx, label="x")
plt.plot(tempy, label="y")
plt.plot(xp, label="xp")
plt.plot(xf, label="xf")
plt.plot(xf, label="xs")
legend = ax.legend(loc='upper left')
plt.show()

s2 = dlm(mu0,SIGMA0,A,R,PHI,Q)
s2.EM(tempy, np.array([ns[0]]),estPHI=True,maxit=100, tol=1e-4,verbose=False)
s2.mu0, s2.SIGMA0, s2.R, s2.PHI, s2.Q

#filter and forecast
x_filter, P_filter = s2.onestep_filter(tempy[ns[0]-1,:],xp[ns[0]-1,:],pp[ns[0]-1,:,:],A[ns[0]-1,:,:])
x_forecast, P_forecast = s2.onestep_forecast(x_filter,P_filter)

#EM fitting
bigA = np.ones([bigymat.shape[0], 1, 1])
dlm1 = dlm(mu0,SIGMA0,bigA,R,PHI,Q)
dlm1.EM(bigymat, ntimes, estPHI=True, maxit=100, tol=1e-4,verbose=True)
dlm1.mu0, dlm1.SIGMA0, dlm1.R, dlm1.PHI, dlm1.Q

dlm1 = dlm(mu0,SIGMA0,bigA,R,PHI,Q)
#dlm1._EM(yy,A,mu0,SIGMA0,PHI0,Q0,R0,ntimes,maxit=40, tol=1e-4, estPHI=True)
dlm1.EM(bigymat, ntimes, estPHI=False, maxit=100, tol=1e-4,verbose=True)
dlm1.mu0, dlm1.SIGMA0, dlm1.R, dlm1.PHI, dlm1.Q




####################################################################################
#real data testing

ana1 = pd.read_csv("analysis.csv")

ana1.head()
ana1.shape

#units: ppb both
#reporting prediction errors
ana1.describe()
yy = ana1["sqrty"].values.reshape(8784,1) 
zz = ana1["sqrtz"].values.reshape(8784,1)

plt.plot(yy)
plt.plot(zz)
sum(pd.isnull(yy))
indices = [k for k,val in enumerate(yy) if pd.isnull(yy[k])]

train = 6588
trainid = np.arange(train) #%75
testid = np.arange(train,8784)


##############################################
#DLM1: random walk + noise
#y_t = mu_t + v_t
#mu_t = mu_{t-1} + w_t
train = 6588
trainid = np.arange(train) #%75
testid = np.arange(train,8784)
fullA = np.repeat(1.0,8784).reshape(8784,1,1)


A = np.repeat(1.0,train).reshape(train,1,1)
mu0 = np.array([5.0])
SIGMA0 = np.array([2.0]).reshape(1,1)
R0 = np.array([2.0]).reshape(1,1)
PHI0 = np.array([1.0]).reshape(1,1)
Q0 = np.array([1.0]).reshape(1,1)

 
dlm1 = dlm(mu0,SIGMA0,A,R0,PHI0,Q0)
ntimes = np.array([train])
dlm1.EM(yy[:train,:], ntimes, estPHI=False, maxit=30)
dlm1.mu0, dlm1.SIGMA0, dlm1.R, dlm1.PHI, dlm1.Q
#loglik =  -5786

xp,xf,pp,pf,K_last = dlm1.filtering(yy[:train,:])
diff = 0
thisxf = xf[train-1,:]
thispf = pf[train-1,:,:]
for i in range(len(testid)):
    thisx,thisp = dlm1.onestep_forecast(thisxf,thispf)
    if not pd.isnull(yy[train]):
        diff += np.sum(np.abs(yy[train,:] - np.dot(fullA[train,:,:],thisx) ))
    thisxf,thispf = dlm1.onestep_filter(yy[train,:],thisx,thisp,fullA[train,:,:])
    train += 1

diff / len(testid)
#error: 0.3980



#####################################
#DLM2: local linear trend
#y_t = mu_t + v_t
#mu_t = mu_{t-1} + b_{t-1} + w1_t
#b_t = b_{t-1} + w2_t
train = 6588
trainid = np.arange(train) #%75
testid = np.arange(train,8784)
fullA = np.ones([8784,1,2])
fullA[:,:,1] = 0.0
 
A = np.ones([train,1,2])
A[:,:,0] = 1.0
A[:,:,1] = 0.0
mu0 = np.array([5.0,1.0])
SIGMA0 = np.array([1.0,0.0,0.0,1.0]).reshape(2,2)
PHI0 = np.array([1.0,1.0,0.0,1.0]).reshape(2,2)
Q0 = np.array([0.3,0,0,0.3]).reshape(2,2)
R0 = np.array([2.0]).reshape(1,1)
ntimes = np.array([train])

dlm2 = dlm(mu0,SIGMA0,A,R0,PHI0,Q0)
dlm2.EM(yy[trainid,:], ntimes, estPHI=False, maxit=30, tol=1e-3)
dlm2.R, dlm2.PHI, dlm2.Q
#loglik = 1644947

xp,xf,pp,pf,K_last = dlm2.filtering(yy[trainid,:])
diff = 0
thisxf = xf[train-1,:]
thispf = pf[train-1,:]
for i in range(len(testid)):
    thisx,thisp = dlm2.onestep_forecast(thisxf,thispf)
    if not pd.isnull(yy[train]):
        diff += np.sum(np.abs(yy[train,:] - np.dot(fullA[train,:,:],thisx) ))
    thisxf,thispf = dlm2.onestep_filter(yy[train,:],thisx,thisp,fullA[train,:,:])
    train += 1

diff / len(testid)
#error: 0.467

##########################
#can be viewed as a regression with time-varying coefficients.
#y_t = a_1t + a_2t x_t + e_t
#(a_1t,a_2t) = G_t %*% (a_{1,t-1},a_{2,t-1})+w_t

#################################
#DLM3: seasonal trend ( 1 harmonic )
#y_t = [1 1 0] %*% [mu_t, s1_t, s1*_t] + v_t
'''
PHI = [1      0       0   ]
      [0      cosw    sinw]
      [0      -sinw   cosw]
'''
train = 6588
trainid = np.arange(train) #%75
testid = np.arange(train,8784)
fullA = np.ones([8784,1,3])
fullA[:,:,2] = 0.0


A = np.ones([train,1,3])
A[:,:,2] = 0.0
mu0 = np.array([10.0,0.0,0.0])
SIGMA0 = np.array([1.0,0.0,0.0,\
                   0.0,1.0,0.0,\
                   0.0,0.0,1.0]).reshape(3,3)
PHI0 = np.array([1.0,0.0,0.0,\
                 0.0,np.cos(2*np.pi/24),np.sin(2*np.pi/24),\
                 0.0,-np.sin(2*np.pi/24),np.cos(2*np.pi/24)]).reshape(3,3)
Q0 = np.array([1.0,0.0,0.0,\
                   0.0,1.0,0.0,\
                   0.0,0.0,1.0]).reshape(3,3)
R0 = np.array([1.0]).reshape(1,1)
ntimes = np.array([train])

dlm3 = dlm(mu0,SIGMA0,A,R0,PHI0,Q0)
dlm3.EM(yy[trainid,:],ntimes,maxit=30, tol=1e-4,estPHI=False)
dlm3.R,dlm3.PHI,dlm3.Q
#loglik: 5603

xp,xf,pp,pf,K_last = dlm3.filtering(yy[trainid,:])
diff = 0
thisxf = xf[train-1,:]
thispf = pf[train-1,:]
for i in range(len(testid)):
    thisx,thisp = dlm3.onestep_forecast(thisxf,thispf)
    if not pd.isnull(yy[train]):
        diff += np.sum(np.abs(yy[train,:] - np.dot(fullA[train,:,:],thisx) ))
    thisxf,thispf = dlm3.onestep_filter(yy[train,:],thisx,thisp,fullA[train,:,:])
    train += 1

diff / len(testid)
#error: 0.4008
    
    
#################################
#DLM4: WITH EXOGENOUS INPUT
#exogenous input with time-varying coefficient
##y_t = [1 1 0 no2_t] %*% [mu_t, s1_t, s1*_t, beta] + v_t

train = 6588
trainid = np.arange(train) #%75
testid = np.arange(train,8784)

fullA = np.ones([8784,1,4])
fullA[:,:,2] = 0.0
fullA[:,:,3] = zz
 

A = np.ones([train,1,4])
A[:,:,2] = 0.0
A[:,:,3] = zz[:train,:]

mu0 = np.array([10.0,0.0,0.0,0.0])
SIGMA0 = np.array([1.0,0.0,0.0,0.0,\
                   0.0,1.0,0.0,0.0,\
                   0.0,0.0,1.0,0.0,\
                   0.0,0.0,0.0,1.0]).reshape(4,4)
PHI0 = np.array([1.0,0.0,0.0,0.0,\
                 0.0,np.cos(2*np.pi/24),np.sin(2*np.pi/24),0.0,\
                 0.0,-np.sin(2*np.pi/24),np.cos(2*np.pi/24),0.0,\
                 0.0,0.0,0.0,1.0]).reshape(4,4)
Q0 = np.array([1.0,0.0,0.0,0.0,\
                   0.0,1.0,0.0,0.0,\
                   0.0,0.0,1.0,0.0,\
                   0.0,0.0,0.0,1.0]).reshape(4,4)
R0 = np.array([2.0]).reshape(1,1)
#BETA0 = np.array([0.1]).reshape(1,1)
#BETA IS INCORPORATED AS A TIME-VARYING COEFFICIENT
ntimes = np.array([train])

dlm4 = dlm(mu0,SIGMA0,A,R0,PHI0,Q0)
dlm4.EM(yy[trainid,:],ntimes,maxit=30)
dlm4.R,dlm4.PHI,dlm4.Q
#loglik =  9314


xp,xf,pp,pf,K_last = dlm4.filtering(yy[trainid,:])
diff = 0
thisxf = xf[train-1,:]
thispf = pf[train-1,:]
for i in range(len(testid)):
    thisx,thisp = dlm4.onestep_forecast(thisxf,thispf)
    if not pd.isnull(yy[train]):
        diff += np.sum(np.abs(yy[train,:] - np.dot(fullA[train,:,:],thisx) ))
    thisxf,thispf = dlm4.onestep_filter(yy[train,:],thisx,thisp,fullA[train,:,:])
    train += 1

diff / len(testid)
#0.2694

#save the results for plot in R
final = dlm(mu0,SIGMA0,fullA,R0,PHI0,Q0)
ntimes = np.array([8784])
final.EM(yy, ntimes, estPHI=False,maxit=30)
final.R,final.PHI,final.Q
 
#smooth the series
xs,ps,pcov,xp,pp = final.smoothing(yy)

series_smooth = xs[:,0] + xs[:,1] + zz.ravel() * xs[:,3]
series_upper = np.zeros(8784) 
series_lower = np.zeros(8784) 
multvec = np.array([1,1,0,0.5])

for i in range(8784):
    multvec[3] = zz[i]
    thiscov = np.dot(np.dot(multvec.T, ps[i,:,:]),multvec)
    series_upper[i] = series_smooth[i] + 1.96 * thiscov
    series_lower[i] = np.maximum(series_smooth[i] - 1.96 * thiscov,0)
    
plt.plot(yy)
plt.plot(series_smooth)

plt.plot(series_upper)
plt.plot(series_lower)

#smooth the trends
trendmu = xs[:,0]
trendmu_upper = np.zeros(8784)
trendmu_lower = np.zeros(8784)

for i in range(8784):
    thisvar = ps[i,0,0]
    trendmu_upper[i] = trendmu[i] + 1.96 * thisvar
    trendmu_lower[i] = np.maximum(trendmu[i] - 1.96 * thisvar,0)

plt.plot(trendmu)
plt.plot(trendmu_upper)
plt.plot(trendmu_lower)

trendseason = xs[:,1]
trendseason_upper = np.zeros(8784)
trendseason_lower = np.zeros(8784)

for i in range(8784):
    thisvar = ps[i,1,1]
    trendseason_upper[i] = trendseason[i] + 1.96 * thisvar
    trendseason_lower[i] = trendseason[i] - 1.96 * thisvar

plt.plot(trendseason)
plt.plot(trendseason_upper)
plt.plot(trendseason_lower)

trendno2 = xs[:,3]
trendno2_upper = np.zeros(8784)
trendno2_lower = np.zeros(8784)

for i in range(8784):
    thisvar = ps[i,3,3]
    trendno2_upper[i] = trendno2[i] + 1.96 * thisvar
    trendno2_lower[i] = trendno2[i] - 1.96 * thisvar

plt.plot(trendno2)
plt.plot(trendno2_upper)
plt.plot(trendno2_lower)

dictionary = {'date':ana1['date'],
              'time':ana1['time'],
              'rawozone':ana1['y'],
              'rawno2':ana1['z'],
              'series_smooth':series_smooth,
              'series_upper':series_upper,
              'series_lower':series_lower,
              'trendmu':trendmu,
              'trendmu_upper':trendmu_upper,
              'trendmu_lower':trendmu_lower,
              'trendseason':trendseason,
              'trendseason_upper':trendseason_upper,
              'trendseason_lower':trendseason_lower,
              'trendno2':trendno2,
              'trendno2_upper':trendno2_upper,
              'trendno2_lower':trendno2_lower}
data = pd.DataFrame(dictionary)



#########################################################################
#################################################################
#bivariate y, univariate x
xdict = dict()
ydict = dict()
nsubj = 20
ns = np.zeros(nsubj, dtype=np.int)


mu0 = np.array([0.0])
SIGMA0 = np.array([1.0]).reshape(1,1)
R = np.array([2.5,0,0,1.5]).reshape(2,2)
PHI = np.array([0.1]).reshape(1,1)
Q = np.array([0.8]).reshape(1,1)


for i in range(nsubj):
    ns[i] = int(np.random.uniform(200,400,1))
    A = np.ones([ns[i],2,1])
    s1 = dlm(mu0,SIGMA0,A,R,PHI,Q)
    tempx, tempy = s1.simulate(ns[i])
    ydict.update({i: tempy})
    xdict.update({i: tempx})

bigxmat = xdict[0]
bigns = [ns[0]]
bigymat = ydict[0]

for i in range(1,nsubj):
    bigxmat = np.vstack((bigxmat, xdict[i]))
    bigymat = np.vstack((bigymat, ydict[i]))
    bigns.append(ns[i])
    
    
ntimes = np.array(bigns) 

#check the first series
tempy = ydict[0]
tempx = xdict[0]
tempy[50:55,:] = None
A = np.ones([ns[0],2,1])
s1 = dlm(mu0,SIGMA0,A,R,PHI,Q)
xp,xf,pp,pf,K_last = s1.filtering(tempy)
xs,ps,pcov,xp,pp = s1.smoothing(tempy)

#plots
fig, ax = plt.subplots()
plt.plot(tempx, label="x")
plt.plot(tempy, label="y")
plt.plot(xp, label="xp")
plt.plot(xf, label="xf")
plt.plot(xf, label="xs")
legend = ax.legend(loc='upper left')
plt.show()

s2 = dlm(mu0,SIGMA0,A,R,PHI,Q)
s2.EM(tempy, np.array([ns[0]]),estPHI=True,maxit=100, tol=1e-4,verbose=False)
s2.mu0, s2.SIGMA0, s2.R, s2.PHI, s2.Q

#filter and forecast
x_filter, P_filter = s2.onestep_filter(tempy[ns[0]-1,:],xp[ns[0]-1,:],pp[ns[0]-1,:,:],A[ns[0]-1,:,:])
x_forecast, P_forecast = s2.onestep_forecast(x_filter,P_filter)

#EM fitting
bigA = np.ones([bigymat.shape[0], 2, 1])
dlm1 = dlm(mu0,SIGMA0,bigA,R,PHI,Q)
dlm1.EM(bigymat, ntimes, estPHI=True, maxit=100, tol=1e-4,verbose=True)
dlm1.mu0, dlm1.SIGMA0, dlm1.R, dlm1.PHI, dlm1.Q

 
#################################################################
#bivariate y, bivariate x
xdict = dict()
ydict = dict()
nsubj = 20
ns = np.zeros(nsubj, dtype=np.int)


mu0 = np.array([0.0,0.0])
SIGMA0 = np.array([1.0,0,0,1.0]).reshape(2,2)
R = np.array([2.5,0,0,1.5]).reshape(2,2)
PHI = np.array([0.1,0,0,0.2]).reshape(2,2)
Q = np.array([0.8,0,0,0.5]).reshape(2,2)


for i in range(nsubj):
    ns[i] = int(np.random.uniform(200,400,1))
    A = np.ones([ns[i],2,2])
    for j in range(ns[i]):
        A[j,:,:] = np.diag([1,1])
    s1 = dlm(mu0,SIGMA0,A,R,PHI,Q)
    tempx, tempy = s1.simulate(ns[i])
    ydict.update({i: tempy})
    xdict.update({i: tempx})

bigxmat = xdict[0]
bigns = [ns[0]]
bigymat = ydict[0]

for i in range(1,nsubj):
    bigxmat = np.vstack((bigxmat, xdict[i]))
    bigymat = np.vstack((bigymat, ydict[i]))
    bigns.append(ns[i])
    
    
ntimes = np.array(bigns) 

#check the first series
tempy = ydict[0]
tempx = xdict[0]
tempy[50:55,:] = None
A = np.ones([ns[0],2,2])
for j in range(ns[0]):
    A[j,:,:] = np.diag([1,1])
s1 = dlm(mu0,SIGMA0,A,R,PHI,Q)
xp,xf,pp,pf,K_last = s1.filtering(tempy)
xs,ps,pcov,xp,pp = s1.smoothing(tempy)

#plots
fig, ax = plt.subplots()
plt.plot(tempx, label="x")
plt.plot(tempy, label="y")
plt.plot(xp, label="xp")
plt.plot(xf, label="xf")
plt.plot(xf, label="xs")
legend = ax.legend(loc='upper left')
plt.show()

s2 = dlm(mu0,SIGMA0,A,R,PHI,Q)
nllk = s2.EM(tempy, np.array([ns[0]]),estPHI=True,maxit=100, tol=1e-4,verbose=False)
s2.mu0, s2.SIGMA0, s2.R, s2.PHI, s2.Q

#filter and forecast
x_filter, P_filter = s2.onestep_filter(tempy[ns[0]-1,:],xp[ns[0]-1,:],pp[ns[0]-1,:,:],A[ns[0]-1,:,:])
x_forecast, P_forecast = s2.onestep_forecast(x_filter,P_filter)

#EM fitting
bigA = np.ones([bigymat.shape[0], 2, 2])
for j in range(bigymat.shape[0]):
    bigA[j,:,:] = np.diag([1,1])
dlm1 = dlm(mu0,SIGMA0,bigA,R,PHI,Q)
dlm1.EM(bigymat, ntimes, estPHI=True, maxit=30, tol=1e-4,verbose=True)
dlm1.mu0, dlm1.SIGMA0, dlm1.R, dlm1.PHI, dlm1.Q

 






