__author__ = 'dlyin'

import numpy as np
import math
import pandas as pd
import operator

class KNNLearner(object):

    def __init__(self,k, method = 'mean'):
        #pass # move along, these aren't the drones you're looking for
        self.k = k
        self.method = method.lower()
        #return k

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.Xtrain = dataX.copy()
        self.Ytrain = dataY.copy()
        #print self.Xtrain
    def query(self,Xtest):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """

        Yret=np.zeros(len(Xtest))
        for i in range(0,Xtest.shape[0]):
            mydict = {}
            for j in range(0, self.Xtrain.shape[0]):
                euclidean = math.sqrt((self.Xtrain[j][0] -Xtest[i][0])**2 + (self.Xtrain[j][1] -Xtest[i][1])**2)
                mydict.update({euclidean:self.Ytrain[j]})
            sorted_mydict = sorted(mydict.items(), key=operator.itemgetter(0))
            sum=0
            #print sorted_mydict
            for n in range(0,self.k):
                sum+=sorted_mydict[n][1]
            Yret[i] = sum/self.k

        #print 'Yrest', Yret
        return Yret

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
