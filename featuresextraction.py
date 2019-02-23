import numpy as np
import random
import os
from tqdm import tqdm

from utils import *
from timeit import default_timer as timer

class FeaturesMatrixBuilder:
    def __init__(self, dataset, textPrep):
        self.dataset = dataset
        self.textPrep = textPrep

    def encodeSingleWordsPositivityFeature(self, text):
        words = text.split()
        positiveCount = len(set(words) & set(self.dataset.positiveWords))
        negativeCount = len(set(words) & set(self.dataset.negativeWords))
        return int(positiveCount > negativeCount)

    def encodeSingleWordsPositivityFeatures(self, text):
        words = text.split()
        pos = len(set(words) & set(self.dataset.positiveWords))
        neg = len(set(words) & set(self.dataset.negativeWords))
        r = (pos+1)/(neg+1)
        return (r > 0.5, r > 1.0, r > 1.5, r > 2.0, r > 2.5)

    def encodeEachSingleWordPositivityFeatures(self, text):
        pos = []
        neg = []
        for pw in self.dataset.positiveWords:
            pos.append(int(pw in text))
        for nw in self.dataset.negativeWords:
            neg.append(int(nw in text))
        return [*pos, *neg]

    def getScreamingPercentageFeature(self, text):
        count = len([letter for letter in text if letter.isupper()])
        return count / len(text)

    def encodeTopNGramsFeature(self, text, topPosNgrams, topNegNgrams, n = 3):
        trigrams = self.textPrep.splitInNGrams(text, n)
        x_count_pos = []
        x_count_neg = []
        for w in topPosNgrams:
            x_count_pos.append(trigrams.count(w))
        for w in topNegNgrams:
            x_count_neg.append(trigrams.count(w))
        # We try here to give more quadratically more weight to the most frequent ngrams
        weights = np.array(list(reversed(range(1, len(x_count_pos) + 1))))**2
        weightedPosSum = np.dot(np.array(x_count_pos), weights)
        weightedNegSum = np.dot(np.array(x_count_neg), weights)
        return int(weightedPosSum > weightedNegSum)
        # return int(sum(x_count_pos) > sum(x_count_neg))

    def encodeSpoilerFeature(self, text):
        return "spoiler" in text

    def encodeTopNGramsFeatures(self, text, topPosNgrams, topNegNgrams, n = 3):
        trigrams = self.textPrep.splitInNGrams(text, n)
        x_count = []
        for w in topPosNgrams:
            x_count.append(trigrams.count(w))
        for w in topNegNgrams:
            x_count.append(trigrams.count(w))
        return x_count

    def buildTrainingData(self):
        print("Building the training data features matrix (positive reviews then negative ones)")
        t = timer()
        X = []
        y = []
        yVals = [1, 0]
        paths = ["./dataset/train/pos", "./dataset/train/neg"]
        for i in range(len(yVals)):
            for root, dirs, files in os.walk(paths[i]):
                for name in tqdm(files):
                    with open(os.path.join(root, name)) as f:
                        text = self.textPrep.cleanText(f.readline())
                        x1 = self.encodeEachSingleWordPositivityFeatures(text)
                        x2 = self.encodeTopNGramsFeatures(text, self.textPrep.topPos4Grams, self.textPrep.topNeg4Grams, 4)
                        x3 = self.encodeSpoilerFeature(text)
                        X.append([*x1, *x2, x3])
                        y.append(yVals[i])
        
        Z = list(zip(X, y))
        random.shuffle(Z)
        X, y = zip(*Z)
        endTimer(t)
        return (np.array(X), np.array(y))

    def buildTestData(self):
        print("Building the training data features matrix")
        t = timer()
        X = []
        ids = []
        for root, dirs, files in os.walk("./dataset/test"):
            for name in tqdm(files):
                with open(os.path.join(root, name)) as f:
                    text = self.textPrep.cleanText(f.readline())
                    x1 = self.encodeEachSingleWordPositivityFeatures(text)
                    x2 = self.encodeTopNGramsFeatures(text, self.textPrep.topPos4Grams, self.textPrep.topNeg4Grams, 4)
                    x3 = self.encodeSpoilerFeature(text)
                    X.append([*x1, *x2, x3])
                ids.append(name.replace(".txt", ""))
        endTimer(t)
        return (np.array(X), np.array(ids))