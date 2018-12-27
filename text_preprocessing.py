#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import preprocessor as p

    
def saveToFile(liste, filename):
    """
    Save a list into a file
    """
    
    with open(filename, "w") as f:
        for el in liste:
            f.write(str(el)+"\n")

def cleanBackslashs(s):
    """
    Most important question :  is "\\" interpreted as "\\" or as "\"?
    """
    return s.replace("\\", "")


p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION)
table = str.maketrans('', '', r"!#$%&'()*+,-./:;<=>?@[\]^_`{|}~")
stop_words = set(stopwords.words('english'))
porter = PorterStemmer()


def cleanTweet(tweet):
    """
    The preprocessing function:
    Clean backslashs
    Clean hashtags, emojis, urls
    Clean RT
    Clean punctuation
    Lower characters
    Delete words too short or too long
    Delete stopwords
    Stem words
    """
    
    res = p.clean(cleanBackslashs(tweet.encode("utf-8").decode("unicode-escape")))
    if(res[:4] == "RT :"):
        res = res[5:]
    
    stripped = [w.translate(table) for w in res.split()]
    return " ".join([porter.stem(w.lower()) for w in stripped if 2 < len(w) < 15 and w.isalnum() and not w in stop_words])


def loadFromFile(filename, datatype=str):
    """
    Load a list from a file
    """
    
    l = []
    with open(filename, "r") as f:
        l = list(map(datatype, f.read().splitlines()))
    return l


def build_dict(filename):
    """
    Given a docset saved in "filename", build the vocabulary associated
    Lower words
    Stem words
    Delete words of anormal length
    Delete not-alpha words
    Delete stop words
    Delete not-frequent words
    """
    
    l = loadFromFile(filename)
    
    words = []
    for doc in l:
        words += doc.split()
    
    unique_words = set(words)
    
    d = dict()
    for w in words:
        d[w] = 1 if not w in d else d[w]+1
    
    cleaned_words = [porter.stem(w.lower()) for w in unique_words if 2 < len(w) < 15 and w.isalpha() and not w in stop_words and d[w]>5]
    
    print(len(cleaned_words))
    
    saveToFile(cleaned_words, filename+"_dict")