from timeit import default_timer as timer
import numpy as np

def endTimer(t):
    print("Took {0} s\n".format(round(timer() - t, 3)))

def getConfusionMatrix(y, yhat):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(y)):
        if y[i] == 1 and yhat[i] == 1:
            tp += 1
        if y[i] == 0 and yhat[i] == 1:
            fp += 1
        if y[i] == 0 and yhat[i] == 0:
            tn += 1
        if y[i] == 1 and yhat[i] == 0:
            fn += 1
    return np.array([[tp, fp],[fn, tn]])

def getMetrics(m):
    precision = m[0,0] / (m[0,0] + m[0,1])
    recall = m[0,0] / (m[0,0] + m[1,0])
    f1 = 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)

def printResults(m):
    print("Confusion matrix([[tp, fp],[fn, tn]]):")
    print(m)
    print("\nMetrics (precision, recall, F1):")
    print(getMetrics(m))
    return

timer = timer