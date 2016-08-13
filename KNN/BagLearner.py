import numpy as np
import math
import pandas as pd
import operator
import random
import LinRegLearner as lrl
import KNNLearner as knn


class BagLearner(object):


    def __init__(self, learner, kwargs={"k",3}, bags=20, boost = 'False'):
        #pass # move along, these aren't the drones you're looking for

        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.learners = []

            #kwargs = {"k":10}
        for i in range(0,bags):
            self.learners.append(learner(**kwargs))


    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        #self.learners = [self.learner(**self.kwargs) for _ in range(self.bags)]
        self.trainX= dataX.copy()
        self.trainY = dataY.copy()


        M = self.trainX.shape[0]


        idxs = [np.random.random_integers(0, M - 1, size=M) for _ in range(self.bags)]
        #self.nbags = map(self.trainX,idxs )
        #print idxs
        #print range(self.bags)
        #print range(self.trainX.shape[0])
        nbagsX = [self.trainX[i] for i in idxs]
        np.array(nbagsX)
        nbagsY = [self.trainY[i] for i in idxs]
        np.array(nbagsY)
        #print 'bags1', nbagsX[0]

        for i in range(self.bags):
            self.learners[i].addEvidence(nbagsX[i], nbagsY[i])

    def query(self,Xtest):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        #learner = lrl.LinRegLearner() # create a LinRegLearner
        #learner = knn.KNNLearner(k=3) #constructor
        Yret=np.zeros(len(Xtest))

        for i in range(self.bags):
            learner = self.learners[i]

            predY = learner.query(Xtest) # get the predictions
            Yret += predY
        #print 'Yrest', Yret
        return Yret/self.bags

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
