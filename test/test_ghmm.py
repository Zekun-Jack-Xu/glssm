import os
import numpy as np

from glssm import *
#######################################################################
#univariate independent example 2 states
nstate = 2
nsubj = 20
xdict = dict()
ydict = dict()
statedict = dict()
ns = np.zeros(nsubj, dtype=np.int)

prior = np.array([0.5,0.5],dtype=np.float)
tpm = np.array([0.9,0.1,0.2,0.8],dtype=np.float).reshape(2,2)
covcube = np.array([1.0,2.0],dtype=np.float).reshape(2,1,1)
coefcube = np.array([0,0.5,0.3,0.2,
                     10,-0.1,-0.2,-0.3]).reshape(2,1,4)
coefcube[0,:,:]

#generate the covariates
for i in range(nsubj):
    ns[i] = int(np.random.uniform(100,200,1))
    twodarray = np.random.multivariate_normal(np.array([0,1.3,0]),
                                              np.array([2.1,0.5,0.8,
                                                        0.5,1.3,0.2,
                                                        0.8,0.2,1]).reshape(3,3),
                                              ns[i])
    tempx = np.zeros([ns[i],4])
    tempx[:,0] = np.ones(ns[i])
    tempx[:,1:] = twodarray
    xdict.update({i: tempx})


#generate the series and states
obj = dtghmm(2, False, prior, tpm, covcube, coefcube)
for i in range(nsubj):
    tempy, tempstate = obj.simulate(ns[i], xdict[i])
    ydict.update({i: tempy})
    statedict.update({i: tempstate})


#gather the dictionary into matrices
bigxmat = xdict[0]
bigns = [ns[0]]
bigymat = ydict[0]
bigstate = list(statedict[0])

for i in range(1,nsubj):
    bigxmat = np.vstack((bigxmat, xdict[i]))
    bigymat = np.vstack((bigymat, ydict[i]))
    bigns.append(ns[i])
    bigstate = bigstate + list(statedict[i])
    
ntimes = np.array(bigns)   

#initial values
prior1 = np.array([0.6,0.4],dtype=np.float)
tpm1 = np.array([0.8,0.2,0.3,0.7],dtype=np.float).reshape(2,2)
covcube1 = np.array([1.3,2.1],dtype=np.float).reshape(2,1,1)
coefcube1 = np.array([0,0.1,0.1,0.1,
                     7,-0.1,-0.1,-0.1]).reshape(2,1,4)
obj = dtghmm(2, False, prior1, tpm1, covcube1, coefcube1)


#fitting
nllk = obj.fit(ydict[0], np.array([ntimes[0]]), xdict[0], coefshrink = 0,
        covshrink = 0, maxit = 100, tol = 1e-4)

obj = dtghmm(2, False, prior1, tpm1, covcube1, coefcube1)
nllk = obj.fit(bigymat, ntimes, bigxmat, coefshrink = 0.0001,
        covshrink = 0.0001, maxit = 100, tol = 1e-4)

obj.prior, obj.tpm, obj.coefcube, obj.covcube

#viterbi
state = obj.viterbi(ydict[0], xdict[0])
#posterior
posterior = obj.smooth(ydict[0], xdict[0])
#predict
obs, probs = obj.predict_1step(ydict[0], xdict[0])
  


###############################################################
#univariate without covariates 2 states --- gaussian mixture model

nstate = 2
nsubj = 20
xdict = dict()
ydict = dict()
statedict = dict()
ns = np.zeros(nsubj, dtype=np.int)

prior = np.array([0.5,0.5],dtype=np.float)
tpm = np.array([1,0,0,1.0],dtype=np.float).reshape(2,2)
covcube = np.array([1.0,2.0],dtype=np.float).reshape(2,1,1)
coefcube = np.array([0.0,10]).reshape(2,1,1)
coefcube[0,:,:]

#generate the covariates
for i in range(nsubj):
    ns[i] = int(np.random.uniform(100,200,1))
    tempx = np.ones([ns[i],1])
    xdict.update({i: tempx})


#generate the series and states
obj = dtghmm(2, False, prior, tpm, covcube, coefcube)
for i in range(nsubj):
    tempy, tempstate = obj.simulate(ns[i], xdict[i])
    ydict.update({i: tempy})
    statedict.update({i: tempstate})


#gather the dictionary into matrices
bigxmat = xdict[0]
bigns = [ns[0]]
bigymat = ydict[0]
bigstate = list(statedict[0])

for i in range(1,nsubj):
    bigxmat = np.vstack((bigxmat, xdict[i]))
    bigymat = np.vstack((bigymat, ydict[i]))
    bigns.append(ns[i])
    bigstate = bigstate + list(statedict[i])
    
ntimes = np.array(bigns)   

#initial values
prior1 = np.array([0.6,0.4],dtype=np.float)
tpm1 = np.array([1,0,0,1.0],dtype=np.float).reshape(2,2)
covcube1 = np.array([1.3,2.1],dtype=np.float).reshape(2,1,1)
coefcube1 = np.array([-6,8.2]).reshape(2,1,1)
obj = dtghmm(2, False, prior1, tpm1, covcube1, coefcube1)


#fitting
nllk = obj.fit(bigymat, ntimes, bigxmat, coefshrink = 0,
        covshrink = 0.0001, maxit = 100, tol = 1e-8)

obj.prior, obj.tpm, obj.coefcube, obj.covcube

#viterbi
state = obj.viterbi(ydict[0], xdict[0])
#posterior
posterior = obj.smooth(ydict[0], xdict[0])
#predict
obs, probs = obj.predict_1step(ydict[0], xdict[0])
  


########################################################################
#univariate AR1 example 2 states
nstate = 2
nsubj = 20
xdict = dict()
ydict = dict()
statedict = dict()
ns = np.zeros(nsubj, dtype=np.int)

prior = np.array([0.5,0.5],dtype=np.float)
tpm = np.array([0.9,0.1,0.2,0.8],dtype=np.float).reshape(2,2)
covcube = np.array([1.0,2.0],dtype=np.float).reshape(2,1,1)
coefcube = np.array([0,0.1,
                     5,0.2]).reshape(2,1,2)
 
 
#generate the series and states
obj = dtghmm(2, True, prior, tpm, covcube, coefcube)
for i in range(nsubj):
    ns[i] = int(np.random.uniform(100,200,1))
    tempy, tempstate = obj.simulate(ns[i])
    ydict.update({i: tempy})
    statedict.update({i: tempstate})


bigns = [ns[0]]
bigymat = ydict[0]
bigstate = list(statedict[0])

for i in range(1,nsubj):
    bigymat = np.vstack((bigymat, ydict[i]))
    bigns.append(ns[i])
    bigstate = bigstate + list(statedict[i])
    
ntimes = np.array(bigns)   


prior1 = np.array([0.7,0.3],dtype=np.float)
tpm1 = np.array([0.85,0.15,0.25,0.75],dtype=np.float).reshape(2,2)
covcube1 = np.array([0.8,2.2],dtype=np.float).reshape(2,1,1)
coefcube1 = np.array([-0.1,0.05,
                     4.3,0.15]).reshape(2,1,2) 


obj = dtghmm(2, True, prior1, tpm1, covcube1, coefcube1)
nllk = obj.fit(bigymat, ntimes, coefshrink = 0.01,
        covshrink = 0.0001, maxit = 100, tol = 1e-4)

obj.prior, obj.tpm, obj.coefcube, obj.covcube

#viterbi
state = obj.viterbi(ydict[0])
#posterior
posterior = obj.smooth(ydict[0])
#predict
obs, probs = obj.predict_1step(ydict[0])

#######################################################################
#multivariate independent example 2 states no covariates
nstate = 2
nsubj = 20
xdict = dict()
ydict = dict()
statedict = dict()
ns = np.zeros(nsubj, dtype=np.int)

prior = np.array([0.5,0.5],dtype=np.float)
tpm = np.array([0.9,0.1,0.2,0.8],dtype=np.float).reshape(2,2)
covcube = np.array([1,0.1,0.1,0.1,1,0.1,0.1,0.1,1,
                    2,0,0,0,3,0,0,0,4]).\
reshape(2,3,3)
coefcube = np.array([-10,-5.0,0,5,10,20]).reshape(2,3,1)
coefcube[0,:,:]

#generate the covariates
for i in range(nsubj):
    ns[i] = int(np.random.uniform(100,200,1))
    tempx = np.ones([ns[i],1])
    xdict.update({i: tempx})


#generate the series and states
obj = dtghmm(2, False, prior, tpm, covcube, coefcube)
for i in range(nsubj):
    tempy, tempstate = obj.simulate(ns[i], xdict[i])
    ydict.update({i: tempy})
    statedict.update({i: tempstate})

 
 
#gather the dictionary into matrices
bigxmat = xdict[0]
bigns = [ns[0]]
bigymat = ydict[0]
bigstate = list(statedict[0])

for i in range(1,nsubj):
    bigxmat = np.vstack((bigxmat, xdict[i]))
    bigymat = np.vstack((bigymat, ydict[i]))
    bigns.append(ns[i])
    bigstate = bigstate + list(statedict[i])
    
ntimes = np.array(bigns)   

#initial values
prior1 = np.array([0.6,0.4],dtype=np.float)
tpm1 = np.array([0.6,0.4,0.3,0.7],dtype=np.float).reshape(2,2)
covcube1 = np.array([1,0.1,0.1,0.1,1,0.1,0.1,0.1,1,
                    2,0,0,0,3,0,0,0,4]).\
reshape(2,3,3)
coefcube1 = np.array([-20,-15.0,-7,8,12,22]).reshape(2,3,1)
obj = dtghmm(2, False, prior1, tpm1, covcube1, coefcube1)


#fitting
nllk = obj.fit(bigymat, ntimes, bigxmat, coefshrink = 0,
        covshrink = 0.0001, maxit = 100, tol = 1e-8)

obj.prior, obj.tpm, obj.coefcube, obj.covcube

#viterbi
state = obj.viterbi(ydict[0], xdict[0])
#posterior
posterior = obj.smooth(ydict[0], xdict[0])
#predict
obs, probs = obj.predict_1step(ydict[0], xdict[0])

##############################################################
#multivariate ar1 example 2 states
nstate = 2
nsubj = 20
xdict = dict()
ydict = dict()
statedict = dict()
ns = np.zeros(nsubj, dtype=np.int)

prior = np.array([0.5,0.5],dtype=np.float)
tpm = np.array([0.9,0.1,0.2,0.8],dtype=np.float).reshape(2,2)
covcube = np.array([1,0.1,0.1,0.1,1,0.1,0.1,0.1,1,
                    2,0,0,0,3,0,0,0,4]).\
reshape(2,3,3)
coefcube = np.array([-10,0.1,0,0,
                     -5, 0,0.1,0,
                     0,  0, 0, 0.1,
                     5, 0.2, 0, 0,
                     10, 0, -0.1, 0,
                     15, 0, 0.1, 0.2]).reshape(2,3,4)
coefcube[0,:,:]

#generate the covariates


#generate the series and states
obj = dtghmm(2, True, prior, tpm, covcube, coefcube)
for i in range(nsubj):
    ns[i] = int(np.random.uniform(100,200,1))
    tempy, tempstate = obj.simulate(ns[i])
    ydict.update({i: tempy})
    statedict.update({i: tempstate})

 
 
#gather the dictionary into matrices
bigns = [ns[0]]
bigymat = ydict[0]
bigstate = list(statedict[0])

for i in range(1,nsubj):
    bigymat = np.vstack((bigymat, ydict[i]))
    bigns.append(ns[i])
    bigstate = bigstate + list(statedict[i])
    
ntimes = np.array(bigns)   

#initial values
prior1 = np.array([0.6,0.4],dtype=np.float)
tpm1 = np.array([0.6,0.4,0.3,0.7],dtype=np.float).reshape(2,2)
covcube1 = np.array([1,0.1,0.1,0.1,1,0.1,0.1,0.1,1,
                    2,0,0,0,3,0,0,0,4]).\
reshape(2,3,3)
coefcube1 = np.array([-13,0,0,0.0,
                     -8, 0,0,0,
                     0,  0, 0, 0,
                     9, 0, 0, 0,
                     13, 0, 0, 0,
                     22, 0, 0, 0]).reshape(2,3,4)
obj = dtghmm(2, True, prior1, tpm1, covcube1, coefcube1)


#fitting
nllk = obj.fit(bigymat, ntimes, bigxmat, coefshrink = 0,
        covshrink = 0.0001, maxit = 100, tol = 1e-8)

obj.prior, obj.tpm, obj.coefcube, obj.covcube

#viterbi
state = obj.viterbi(ydict[0])
#posterior
posterior = obj.smooth(ydict[0])
#predict
obs, probs = obj.predict_1step(ydict[0])


#############################################################
#univariate independent example 3 states
nstate = 3
nsubj = 20
xdict = dict()
ydict = dict()
statedict = dict()
ns = np.zeros(nsubj, dtype=np.int)

prior = np.array([0.5,0.3,0.2],dtype=np.float)
tpm = np.array([0.7,0.2,0.1,0.1,0.8,0.1,0.2,0.2,0.6],dtype=np.float).reshape(3,3)
covcube = np.array([1.0,2.0,3.0],dtype=np.float).reshape(3,1,1)
coefcube = np.array([0,0.5,0.3,0.2,
                     5,-0.1,-0.2,-0.3,
                     10,0.3,0.6,0.9]).reshape(3,1,4)
coefcube[0,:,:]

#generate the covariates
for i in range(nsubj):
    ns[i] = int(np.random.uniform(100,200,1))
    twodarray = np.random.multivariate_normal(np.array([0,1.3,0]),
                                              np.array([2.1,0.5,0.8,
                                                        0.5,1.3,0.2,
                                                        0.8,0.2,1]).reshape(3,3),
                                              ns[i])
    tempx = np.zeros([ns[i],4])
    tempx[:,0] = np.ones(ns[i])
    tempx[:,1:] = twodarray
    xdict.update({i: tempx})


#generate the series and states
obj = dtghmm(3, False, prior, tpm, covcube, coefcube)
for i in range(nsubj):
    tempy, tempstate = obj.simulate(ns[i], xdict[i])
    ydict.update({i: tempy})
    statedict.update({i: tempstate})


#gather the dictionary into matrices
bigxmat = xdict[0]
bigns = [ns[0]]
bigymat = ydict[0]
bigstate = list(statedict[0])

for i in range(1,nsubj):
    bigxmat = np.vstack((bigxmat, xdict[i]))
    bigymat = np.vstack((bigymat, ydict[i]))
    bigns.append(ns[i])
    bigstate = bigstate + list(statedict[i])
    
ntimes = np.array(bigns)   

#initial values
prior1 = np.array([0.6,0.3,0.1],dtype=np.float)
tpm1 = np.array([0.7,0.2,0.1,0.1,0.8,0.1,0.2,0.2,0.6],dtype=np.float).reshape(3,3)
covcube1 = np.array([1.0,2.0,3.0],dtype=np.float).reshape(3,1,1)
coefcube1 = np.array([0,0.5,0.3,0.2,
                     5,-0.1,-0.2,-0.3,
                     10,0.3,0.6,0.9]).reshape(3,1,4)
obj = dtghmm(3, False, prior1, tpm1, covcube1, coefcube1)


#fitting
 
nllk = obj.fit(bigymat, ntimes, bigxmat, coefshrink = 0.0001,
        covshrink = 0.0001, maxit = 100, tol = 1e-4)

obj.prior, obj.tpm, obj.coefcube, obj.covcube

#viterbi
state = obj.viterbi(ydict[0], xdict[0])
#posterior
posterior = obj.smooth(ydict[0], xdict[0])
#predict
obs, probs = obj.predict_1step(ydict[0], xdict[0])
  

#############################################################
#univariate independent example 3 states nocovariates

nstate = 3
nsubj = 20
xdict = dict()
ydict = dict()
statedict = dict()
ns = np.zeros(nsubj, dtype=np.int)

prior = np.array([0.3,0.3,0.4])
tpm = np.array([0.7,0.2,0.1,0.1,0.8,0.1,0.2,0.2,0.6],dtype=np.float).reshape(3,3)
covcube = np.array([1.0,2.0,3.0],dtype=np.float).reshape(3,1,1)
coefcube = np.array([-8,0,7]).reshape(3,1,1)

#generate the covariates
for i in range(nsubj):
    ns[i] = int(np.random.uniform(100,200,1))
    tempx = np.ones([ns[i],1])
    xdict.update({i: tempx})


#generate the series and states
obj = dtghmm(3, False, prior, tpm, covcube, coefcube)
for i in range(nsubj):
    tempy, tempstate = obj.simulate(ns[i], xdict[i])
    ydict.update({i: tempy})
    statedict.update({i: tempstate})


#gather the dictionary into matrices
bigxmat = xdict[0]
bigns = [ns[0]]
bigymat = ydict[0]
bigstate = list(statedict[0])

for i in range(1,nsubj):
    bigxmat = np.vstack((bigxmat, xdict[i]))
    bigymat = np.vstack((bigymat, ydict[i]))
    bigns.append(ns[i])
    bigstate = bigstate + list(statedict[i])
    
ntimes = np.array(bigns)   

#initial values

prior1 = np.array([0.3,0.3,0.4])
tpm1 = np.array([0.7,0.2,0.1,0.1,0.8,0.1,0.2,0.2,0.6],dtype=np.float).reshape(3,3)
covcube1 = np.array([1.0,2.0,3.0],dtype=np.float).reshape(3,1,1)
coefcube1 = np.array([-5,0,5.0]).reshape(3,1,1)

obj = dtghmm(3, False, prior1, tpm1, covcube1, coefcube1)


#fitting
nllk = obj.fit(bigymat, ntimes, bigxmat, coefshrink = 0,
        covshrink = 0.0001, maxit = 100, tol = 1e-8)

obj.prior, obj.tpm, obj.coefcube, obj.covcube

#viterbi
state = obj.viterbi(ydict[0], xdict[0])
#posterior
posterior = obj.smooth(ydict[0], xdict[0])
#predict
obs, probs = obj.predict_1step(ydict[0], xdict[0])
  

###################################################
#univariate ar1 example 3 states
nstate = 2
nsubj = 20
xdict = dict()
ydict = dict()
statedict = dict()
ns = np.zeros(nsubj, dtype=np.int)

prior = np.array([0.3,0.3,0.4])
tpm = np.array([0.7,0.2,0.1,0.1,0.8,0.1,0.2,0.2,0.6],dtype=np.float).reshape(3,3)
covcube = np.array([1.0,2.0,3.0],dtype=np.float).reshape(3,1,1)
coefcube = np.array([-8,0.1,0,0,7,-0.1]).reshape(3,1,2)

 
#generate the series and states
obj = dtghmm(3, True, prior, tpm, covcube, coefcube)
for i in range(nsubj):
    ns[i] = int(np.random.uniform(100,200,1))
    tempy, tempstate = obj.simulate(ns[i])
    ydict.update({i: tempy})
    statedict.update({i: tempstate})


bigns = [ns[0]]
bigymat = ydict[0]
bigstate = list(statedict[0])

for i in range(1,nsubj):
    bigymat = np.vstack((bigymat, ydict[i]))
    bigns.append(ns[i])
    bigstate = bigstate + list(statedict[i])
    
ntimes = np.array(bigns)   


prior1 = np.array([0.3,0.3,0.4])
tpm1 = np.array([0.7,0.2,0.1,0.1,0.8,0.1,0.2,0.2,0.6],dtype=np.float).reshape(3,3)
covcube1 = np.array([1.0,2.0,3.0],dtype=np.float).reshape(3,1,1)
coefcube1 = np.array([-8,0.1,0,0,7,-0.1]).reshape(3,1,2)

 
obj = dtghmm(3, True, prior1, tpm1, covcube1, coefcube1)
nllk = obj.fit(bigymat, ntimes, coefshrink = 0.01,
        covshrink = 0.0001, maxit = 100, tol = 1e-4)

obj.prior, obj.tpm, obj.coefcube, obj.covcube

#viterbi
state = obj.viterbi(ydict[0])
#posterior
posterior = obj.smooth(ydict[0])
#predict
obs, probs = obj.predict_1step(ydict[0])


##################################################
#multivariate independent 3 states no covariates

nstate = 3
nsubj = 20
xdict = dict()
ydict = dict()
statedict = dict()
ns = np.zeros(nsubj, dtype=np.int)

prior = np.array([0.3,0.3,0.4])
tpm = np.array([0.7,0.2,0.1,0.1,0.8,0.1,0.2,0.2,0.6],dtype=np.float).reshape(3,3)
covcube = np.array([1,0,0,1,2,0.0,0,2,3,0,0,3],dtype=np.float).reshape(3,2,2)
coefcube = np.array([-8,-5,0,0,5,8]).reshape(3,2,1)

#generate the covariates
for i in range(nsubj):
    ns[i] = int(np.random.uniform(100,200,1))
    tempx = np.ones([ns[i],1])
    xdict.update({i: tempx})


#generate the series and states
obj = dtghmm(3, False, prior, tpm, covcube, coefcube)
for i in range(nsubj):
    tempy, tempstate = obj.simulate(ns[i], xdict[i])
    ydict.update({i: tempy})
    statedict.update({i: tempstate})


#gather the dictionary into matrices
bigxmat = xdict[0]
bigns = [ns[0]]
bigymat = ydict[0]
bigstate = list(statedict[0])

for i in range(1,nsubj):
    bigxmat = np.vstack((bigxmat, xdict[i]))
    bigymat = np.vstack((bigymat, ydict[i]))
    bigns.append(ns[i])
    bigstate = bigstate + list(statedict[i])
    
ntimes = np.array(bigns)   

#initial values

prior1 = np.array([0.3,0.3,0.4])
tpm1 = np.array([0.7,0.2,0.1,0.1,0.8,0.1,0.2,0.2,0.6],dtype=np.float).reshape(3,3)
covcube1 = np.array([1,0,0,1,2,0.0,0,2,3,0,0,3],dtype=np.float).reshape(3,2,2)
coefcube1 = np.array([-8,-5,0,0,5,8.0]).reshape(3,2,1)

obj = dtghmm(3, False, prior1, tpm1, covcube1, coefcube1)


#fitting
nllk = obj.fit(bigymat, ntimes, bigxmat, coefshrink = 0,
        covshrink = 0.0001, maxit = 100, tol = 1e-8)

obj.prior, obj.tpm, obj.coefcube, obj.covcube

#viterbi
state = obj.viterbi(ydict[0], xdict[0])
#posterior
posterior = obj.smooth(ydict[0], xdict[0])
#predict
obs, probs = obj.predict_1step(ydict[0], xdict[0])
  

########################################
#multivariate ar1 3 states

nstate = 3
nsubj = 20
xdict = dict()
ydict = dict()
statedict = dict()
ns = np.zeros(nsubj, dtype=np.int)

prior = np.array([0.3,0.3,0.4])
tpm = np.array([0.7,0.2,0.1,0.1,0.8,0.1,0.2,0.2,0.6],dtype=np.float).reshape(3,3)
covcube = np.array([1,0,0,1,2,0.0,0,2,3,0,0,3],dtype=np.float).reshape(3,2,2)
coefcube = np.array([-8,0.1,0,
                     -5,0,0.1,
                     0,0,0,0,0,0,
                     5,-0.1,0,
                     8,0,-0.1]).reshape(3,2,3)

 
#generate the series and states
obj = dtghmm(3, True, prior, tpm, covcube, coefcube)
for i in range(nsubj):
    ns[i] = int(np.random.uniform(100,200,1))
    tempy, tempstate = obj.simulate(ns[i])
    ydict.update({i: tempy})
    statedict.update({i: tempstate})


#gather the dictionary into matrices
bigns = [ns[0]]
bigymat = ydict[0]
bigstate = list(statedict[0])

for i in range(1,nsubj):
    bigymat = np.vstack((bigymat, ydict[i]))
    bigns.append(ns[i])
    bigstate = bigstate + list(statedict[i])
    
ntimes = np.array(bigns)   

#initial values


prior1 = np.array([0.3,0.3,0.4])
tpm1 = np.array([0.7,0.2,0.1,0.1,0.8,0.1,0.2,0.2,0.6],dtype=np.float).reshape(3,3)
covcube1 = np.array([1,0,0,1,2,0.0,0,2,3,0,0,3],dtype=np.float).reshape(3,2,2)
coefcube1 = np.array([-8,0.1,0,
                     -5,0,0.1,
                     0,0,0,0,0,0,
                     5,-0.1,0,
                     8,0,-0.1]).reshape(3,2,3)

obj = dtghmm(3, True, prior1, tpm1, covcube1, coefcube1)


#fitting
nllk = obj.fit(bigymat, ntimes, bigxmat, coefshrink = 0,
        covshrink = 0.0001, maxit = 100, tol = 1e-8)

obj.prior, obj.tpm, obj.coefcube, obj.covcube

#viterbi
state = obj.viterbi(ydict[0])
#posterior
posterior = obj.smooth(ydict[0])
#predict
obs, probs = obj.predict_1step(ydict[0])