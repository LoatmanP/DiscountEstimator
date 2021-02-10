from __future__ import division
import time
from sklearn.model_selection import train_test_split
import scipy.optimize
import pickle, math, sys
#from psychopy import gui, core
from scipy.optimize import fmin
import pandas as pd
import pandas
import numpy as np


class ExponentialClassifier(object):
    def __init__(self, discountRate = -5, rho = .01):
        self. discountRate = discountRate
        self.rho = rho 
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    
    def fit(self,X,y):
        ranges = [[-8, 8, .1], [0, 1.1, .1]]
        opt = scipy.optimize.brute(self.sse, ranges, args = (X,y, ))
        self.discountRate = opt[0]
        self.rho = opt[1]
        return self
        
    def sse(self, params, X, y):
        if (params[0] < -7.3) or (params[0] > 10) or (params[1] <= 0.):
            return 100000000000000000000000
        yhat = self.choice(X, params)

        boundary = .00000001
        yhat = np.clip(yhat, boundary, 1-boundary)
        asdf1 = -y * np.log(yhat)
        asdf2 = (1-y) * np.log(1-yhat)
        sse = np.sum(np.power(asdf1 - asdf2, 2))
        #print([params, sse]) 
        return sse
    
    def choice(self, X, params=[]):
        # if params are NOT provided, use internally stored values
        if len(params) == 0:
            rho = self.rho
            discRate = np.exp(self.discountRate)
        # if params ARE provided, use them
        else:
            discRate = np.exp(params[0])
            rho = params[1]

        ssVal = (X[:,0] * numpy.exp((-discRate * X[:,1])))
        llVal = (X[:,2] * numpy.exp((-discRate * X[:,3])))

        pLL = 1 / (1 + np.exp(-rho *(llVal - ssVal)))

        return pLL


    def predict_proba(self,X, params=[]):
        return self.choice(X,params)
        
    def predict(self, X, params=[]):
        return self.choice(X, params).round()
    
    def set_params(self,discountRate=-5, rho = .01):
        self.discountRate = discountRate
        self.rho = rho
        return self
        

    def get_params(self, deep =True):
        return {'discountRate': self.discountRate, 'rho': self.rho} 
       
class HyperbolicClassifier(object):
    def __init__(self, discountRate = -5, rho = .01):
        self. discountRate = discountRate
        self.rho = rho 
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    
    def fit(self,X,y):
        ranges = [[-8, 8, .1], [0, 1.1, .1]]
        opt = scipy.optimize.brute(self.sse, ranges, args = (X,y, ))
        self.discountRate = opt[0]
        self.rho = opt[1]
        return self
        
    def sse(self, params, X, y):
        if (params[0] < -7.3) or (params[0] > 10) or (params[1] <= 0.):
            # return something 'bad' (i.e., big)
            return 10000000000000
        yhat = self.choice(X, params)
        

        boundary = .00000001
        yhat = np.clip(yhat, boundary, 1-boundary)
        
        asdf1 = -y * np.log(yhat)
        asdf2 = (1-y) * np.log(1-yhat)
        sse = np.sum(np.power(asdf1 - asdf2, 2))
        #print([params, sse])
        return sse
    
    def choice(self, X, params=[]):
        # if params are NOT provided, use internally stored values
        if len(params) == 0:
            rho = self.rho
            discRate = np.exp(self.discountRate)
        # if params ARE provided, use them
        else:
            discRate = np.exp(params[0])
            rho = params[1]
        
        ssVal = (X[:,0] *  1/(1+ discRate * X[:,1]))
        llVal = (X[:,2] * 1/(1+ discRate * X[:,3]))

        pLL = 1 / (1 + np.exp(-rho *(llVal - ssVal)))

        return pLL


    def predict_proba(self,X, params=[]):
        return self.choice(X,params)
        
    def predict(self, X, params=[]):
        return self.choice(X, params).round()
    
    
    def set_params(self,discountRate=-5, rho = .01):
        self.discountRate = discountRate
        self.rho = rho
        return self
        
    def get_params(self, deep =True):
        return {'discountRate': self.discountRate, 'rho': self.rho} 


class QuasiHyperbolicClassifier(object):
    def __init__(self, discountRate = -5, rho = .01, bias = .01):
        self. discountRate = discountRate
        self.rho = rho 
        self.bias = bias
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    
    def fit(self,X,y):
        ranges = [[-8, 8, .1], [0, 1.1, .1], [0.1,1.1,.1]]
        opt = scipy.optimize.brute(self.sse, ranges, args = (X,y, ))
        self.discountRate = opt[0]
        self.rho = opt[1]
        self.bias = opt[2]
        return self
        
    def sse(self, params, X, y):
        if (params[0] < -7.3) or (params[0] > 10) or (params[1] <= 0.) or (params[2] > 2) or (params[2] < .000000000001):
            # return something 'bad' (i.e., big)
            return 10000000000000
        yhat = self.choice(X, params)

        boundary = .00000001
        yhat = np.clip(yhat, boundary, 1-boundary)
        
        asdf1 = -y * np.log(yhat)
        asdf2 = (1-y) * np.log(1-yhat)
        sse = np.sum(np.power(asdf1 - asdf2, 2))
        #print([params, sse])
        return sse
    
    def choice(self, X, params=[]):
        # if params are NOT provided, use internally stored values
        if len(params) == 0:
            rho = self.rho
            discRate = np.exp(self.discountRate)
            bias = self.bias
        # if params ARE provided, use them
        else:
            discRate = np.exp(params[0])
            rho = params[1]
            bias = params[2]
        
        if bias == 0:
            ssVal = (X[:,0] * numpy.exp((-discRate * X[:,1])))
            llVal = (X[:,2] * numpy.exp((-discRate * X[:,3])))
        else:
            
            ssVal = (X[:,0] * bias * numpy.exp((-discRate * X[:,1])))
            llVal = (X[:,2] * bias * numpy.exp((-discRate * X[:,3])))

        pLL = 1 / (1 + np.exp(-rho *(llVal - ssVal)))

        return pLL


    def predict_proba(self,X, params=[]):
        return self.choice(X,params)
        
    def predict(self, X, params=[]):
        return self.choice(X, params).round()
    
    def set_params(self,discountRate=-5, rho = .01, bias = .01):
        self.discountRate = discountRate
        self.rho = rho
        self.bias = bias
        return self
        
    def get_params(self, deep =True):
        return {'discountRate': self.discountRate, 'rho': self.rho, 'bias': self.bias} 
        
class GeneralizedHyperbolicClassifier(object):
    def __init__(self, discountRate = -5, rho = .01, curve = .01):
        self. discountRate = discountRate
        self.rho = rho 
        self.curve = curve
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    
    def fit(self,X,y):
        ranges = [[-8, 8, .1], [0, 1.1, .1], [0, 2,.1]]
        opt = scipy.optimize.brute(self.sse, ranges, args = (X,y, ))
        self.discountRate = opt[0]
        self.rho = opt[1]
        self.curve = opt[2]
        return self
        
    def sse(self, params, X, y):
        if (params[0] < -7.3) or (params[0] > 10) or (params[1] <= 0.) or (params[2] > 2) or (params[2] < -2):
            # return something 'bad' (i.e., big)
            return 10000000000000
        yhat = self.choice(X, params)

        boundary = .00000001
        yhat = np.clip(yhat, boundary, 1-boundary)
        
        asdf1 = -y * np.log(yhat)
        asdf2 = (1-y) * np.log(1-yhat)
        sse = np.sum(np.power(asdf1 - asdf2, 2))
        #print([params, sse])
        return sse
    
    def choice(self, X, params=[]):
        # if params are NOT provided, use internally stored values
        if len(params) == 0:
            rho = self.rho
            discRate = np.exp(self.discountRate)
            curve = self.curve
        # if params ARE provided, use them
        else:
            discRate = np.exp(params[0])
            rho = params[1]
            curve = params[2]
        if curve == 0:
            ssVal = (X[:,0] * numpy.exp((-discRate * X[:,1])))
            llVal = (X[:,2] * numpy.exp((-discRate * X[:,3])))
        else:
            ssVal = (X[:,0] * np.power( 1-((-curve)*discRate*X[:,1]), (1/(-curve)))) 
            llVal = (X[:,2] * np.power( 1-((-curve)*discRate*X[:,3]), (1/(-curve))))

        pLL = 1 / (1 + np.exp(-rho *(llVal - ssVal)))

        return pLL


    def predict_proba(self,X, params=[]):
        return self.choice(X,params)
        
    def predict(self, X, params=[]):
        return self.choice(X, params).round()
    
    def set_params(self,discountRate=-5, rho = .01, curve = .01):
        self.discountRate = discountRate
        self.rho = rho
        self.curve = curve
        return self
        
    def get_params(self, deep =True):
        return {'discountRate': self.discountRate, 'rho': self.rho, 'curve': self.curve}
         
class HyperboloidClassifier(object):
    def __init__(self, discountRate = -5, rho = .01, curve = .01):
        self. discountRate = discountRate
        self.rho = rho 
        self.curve = curve
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    
    def fit(self,X,y):
        ranges = [[-8, 8, .1], [0, 1.1, .1], [0.0001,2,.1]]
        opt = scipy.optimize.brute(self.sse, ranges, args = (X,y, ))
        self.discountRate = opt[0]
        self.rho = opt[1]
        self.curve = opt[2]
        return self
        
    def sse(self, params, X, y):
        if (params[0] < -7.3) or (params[0] > 10) or (params[1] <= 0.) or (params[2] > 1.1) or (params[2] < -.1):
            # return something 'bad' (i.e., big)
            return 10000000000000
        yhat = self.choice(X, params)

        boundary = .00000001
        yhat = np.clip(yhat, boundary, 1-boundary)
        
        asdf1 = -y * np.log(yhat)
        asdf2 = (1-y) * np.log(1-yhat)
        sse = np.sum(np.power(asdf1 - asdf2, 2))
        #print([params, sse])
        return sse
    
    def choice(self, X, params=[]):
        # if params are NOT provided, use internally stored values
        if len(params) == 0:
            rho = self.rho
            discRate = np.exp(self.discountRate)
            curve = self.curve
        # if params ARE provided, use them
        else:
            discRate = np.exp(params[0])
            rho = params[1]
            curve = params[2]
            
        ssVal = (X[:,0]/np.power(1+discRate*X[:,1],curve)) 
        llVal = (X[:,2]/np.power(1+discRate*X[:,3],curve)) 

        pLL = 1 / (1 + np.exp(-rho *(llVal - ssVal)))

        return pLL


    def predict_proba(self,X, params=[]):
        return self.choice(X,params)
        
    def predict(self, X, params=[]):
        return self.choice(X, params).round()
    
    def set_params(self,discountRate=-5, rho = .01, curve = .01):
        self.discountRate = discountRate
        self.rho = rho
        self.curve = curve
        return self
        
    def get_params(self, deep =True):
        return {'discountRate': self.discountRate, 'rho': self.rho, 'curve': self.curve} 

class HyperbolicWithExponentClassifier(object):
    def __init__(self, discountRate = -5, rho = .01, curve = .01):
        self. discountRate = discountRate
        self.rho = rho 
        self.curve = curve
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    
    def fit(self,X,y):
        ranges = [[-8, 8, .1], [0, 1.1, .1], [0.0001,2,.1]]
        opt = scipy.optimize.brute(self.sse, ranges, args = (X,y, ))
        self.discountRate = opt[0]
        self.rho = opt[1]
        self.curve = opt[2]
        return self
        
    def sse(self, params, X, y):
        if (params[0] < -7.3) or (params[0] > 10) or (params[1] <= 0.) or (params[2] > 1.1) or (params[2] < -.1):
            # return something 'bad' (i.e., big)
            return 10000000000000
        yhat = self.choice(X, params)

        boundary = .00000001
        yhat = np.clip(yhat, boundary, 1-boundary)
        
        asdf1 = -y * np.log(yhat)
        asdf2 = (1-y) * np.log(1-yhat)
        sse = np.sum(np.power(asdf1 - asdf2, 2))
        #print([params, sse])
        return sse
    
    def choice(self, X, params=[]):
        # if params are NOT provided, use internally stored values
        if len(params) == 0:
            rho = self.rho
            discRate = np.exp(self.discountRate)
            curve = self.curve
        # if params ARE provided, use them
        else:
            discRate = np.exp(params[0])
            rho = params[1]
            curve = params[2]
       
        ssVal = (X[:,0]/(1+discRate*np.power(X[:,1],curve)))
        llVal = (X[:,2]/(1+discRate*np.power(X[:,3],curve)))

        pLL = 1 / (1 + np.exp(-rho *(llVal - ssVal)))

        return pLL


    def predict_proba(self,X, params=[]):
        return self.choice(X,params)
        
    def predict(self, X, params=[]):
        return self.choice(X, params).round()
    
    def set_params(self,discountRate=-5, rho = .01, curve = .01):
        self.discountRate = discountRate
        self.rho = rho
        self.curve = curve
        return self
        
    def get_params(self, deep =True):
        return {'discountRate': self.discountRate, 'rho': self.rho, 'curve': self.curve} 
        



        
