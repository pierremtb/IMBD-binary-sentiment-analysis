# # Mini Project 2: Code
# Alexandre Banon, Vincent Delmas, Pierre Jacquier (Group 90) — COMP551

import numpy as np
from tqdm import tqdm

from datasetloading import Lexicons
from textpreprocessing import TextPreprocessing
from featuresextraction import FeaturesMatrixBuilder
from utils import *
from naivebayes import NaiveBayes

t = timer()

# Loading datasets
dataset = Lexicons().load()

# Warming up textrocessing engines
textPrep = TextPreprocessing(ngrams_n=4, ngrams_count=2000).load(dataset.stopWords)

# Warming up the FeaturesMatrixBuilder
featuresMatrix = FeaturesMatrixBuilder(dataset, textPrep)

# Doing the actual training on the first 22000 reviews
XTrain, yTrain = featuresMatrix.buildTrainingData()
nb = NaiveBayes()
nb.fit(XTrain[:22000,:], yTrain[:22000])

# Validating on the remaining
y = yTrain[22000:]
yhat = nb.predict(XTrain[22000:,:])
m = getConfusionMatrix(yTrain[22000:], yhat)
print("\n=== RESULTS ===")
endTimer(t)
printResults(m)

# Running the model on the test set
print("Training using the whole training set this time")
nb.fit(XTrain, yTrain)
(XTest, ids) = featuresMatrix.buildTestData()
yhat = nb.predict(XTest)
with open("output/test.txt", "w") as f:
    t = timer()
    print("Writing the test results file")
    f.write("Id,Category\n")
    for i, yi in tqdm(enumerate(yhat)):
        f.write("{0},{1}\n".format(ids[i], yi))
    endTimer(t)
    print("DONE.")

