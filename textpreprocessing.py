
import re
import os
import random
from timeit import default_timer as timer
from collections import Counter

from utils import *
from datasetloading import *

#%% [markdown]
# ## Text preprocessing
class TextPreprocessing:
    def __init__(self, ngrams_n = 4, ngrams_count = 200):
        self.ngrams_n = ngrams_n
        self.ngrams_count = ngrams_count
        self.REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
        self.REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    def load(self, stopWords):
        print("Loading text preprocessing engines")
        t = timer()           
        self.stopWords = stopWords 
        # Getting raw grams
        rawTopPos4Grams = self.getFilesTopNGrams("./train/pos", self.ngrams_n, self.ngrams_count)
        rawTopNeg4Grams = self.getFilesTopNGrams("./train/neg", self.ngrams_n, self.ngrams_count)
        # Removing duplicates
        self.topPos4Grams = [x for x in rawTopPos4Grams if not x in rawTopNeg4Grams]
        self.topNeg4Grams = [x for x in rawTopNeg4Grams if not x in rawTopPos4Grams]
        # Writing them in a file for debuging purpose
        with open("output/top_neg_4grams.txt", "w") as f:
            for item in self.topNeg4Grams:
                f.write("%s\n" % item)
        with open("output/top_pos_4grams.txt", "w") as f:
            for item in self.topPos4Grams:
                f.write("%s\n" % item)
        endTimer(t)
        return self

    def cleanText(self, text):
        text = self.REPLACE_NO_SPACE.sub("", text.lower())
        text = self.REPLACE_WITH_SPACE.sub(" ", text)
        word_list = text.split()
        return ' '.join([i for i in word_list if i not in self.stopWords])

    def splitInNGrams(self, text, n):
        ngrams = []
        words = self.cleanText(text).split()
        for i in range(len(words)):
            if i < (len(words) - n + 1):
                ngram = ""
                for k in range(n):
                    ngram += words[i + k]
                    if k < n - 1:
                        ngram += " "
                ngrams.append(ngram)
        return ngrams

    def getFilesTopNGrams(self, directory, n, count):
        trigrams = []
        for root, dirs, files in os.walk(directory):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    trigrams.extend(self.splitInNGrams(f.readline(), n))
        counts = Counter(trigrams).most_common(count)
        return [count[0] for count in counts]