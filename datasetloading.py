import codecs
from timeit import default_timer as timer

from utils import *

class Lexicons:
    def load(self):
        print("Loading lexicons")
        t = timer()
        self.positiveWords = self.readWords("./lexicon/positive-words.txt")
        self.negativeWords = self.readWords("./lexicon/negative-words.txt")
        self.stopWords = ['in','of','at','a','the']
        endTimer(t)
        return self

    def readWords(self, file):
        words = []
        with codecs.open(file, encoding="latin-1") as f:
            for line in f:
                words.append(line.strip())
        return words