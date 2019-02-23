import numpy as np
from tqdm import tqdm

from utils import *

class NaiveBayes():
    theta_1 = 0
    Theta = []

    def fit(self, X, y):
        print("Fitting the Naive Bayes classifier (in two steps)")
        t = timer()
        n = len(y)
        self.theta_1 = len(y[np.where(y == 1)]) / n
        self.Theta = np.zeros((X.shape[1], 2)) 
        sumParams = np.zeros((X.shape[1], 2))
        for i in tqdm(range(len(y))):
            for j in range(X.shape[1]):
                sumParams[j, 1] += (X[i, j] * y[i])
                sumParams[j, 0] += X[i, j] * (1 - y[i])
        
        for j in tqdm(range(X.shape[1])):
            self.Theta[j, 1] = (sumParams[j, 1] + 1) / (self.theta_1 * n + 2)
            self.Theta[j, 0] = (sumParams[j, 0] + 1) / ((1 - self.theta_1) * n + 2)
        endTimer(t)
        return

    def predict(self, X):
        print("Doing the prediction")
        t = timer()
        yhat = []
        for i in tqdm(range(X.shape[0])):
            prob = [1 - self.theta_1, self.theta_1]
            for a in [0, 1]:
                for j in range(X.shape[1]):
                    if X[i, j] == 1:
                        prob[a] *= self.Theta[j, a]
                    else:
                        prob[a] *= (1 - self.Theta[j, a])
            yhat.append(int(prob[1] > prob[0]))
        endTimer(t)
        return yhat