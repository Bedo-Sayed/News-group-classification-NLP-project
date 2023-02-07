import nltk
from nltk.stem import LancasterStemmer
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download("stopwords")
import sklearn
import numpy as np
import re
import string
import pandas as pd
import os
import sys
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from datetime import datetime
from nltk.stem.snowball import SnowballStemmer
from sklearn.datasets import fetch_20newsgroups

def pre_process(df):

    stemmer = SnowballStemmer(language="english")
    stop = set(stopwords.words("english"))
    n = len(df["data"])
    out = []

    for i in range(0 , n):
        tmp = word_tokenize(df["data"][i].lower())
        target = df["target"][i]

        filtered_list = []
        for word in tmp:
            if len(word) > 1 and word.casefold() not in stop:
                s = stemmer.stem(word)
                filtered_list.append(s)

        out.append(' '.join(filtered_list))

    return out