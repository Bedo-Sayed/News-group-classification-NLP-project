import matplotlib.pyplot as plt
import pandas as pd
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
nltk.download('omw-1.4')
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
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import time
import pickle
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

def plot_bar_graphs(training_time , testing_time , score):

    models = ["Logistic Regression" , "SVM" , "Naive bayes"]
    fig = plt.figure(figsize=(10, 5))

    #Training time bar graph
    plt.bar(models, training_time , color='green', width=0.5)
    plt.xlabel("Model name")
    plt.ylabel("Training time in seconds")
    plt.title("Training time")
    plt.show()

    #Testing time bar graph
    plt.bar(models, testing_time, color='green', width=0.5)
    plt.xlabel("Model name")
    plt.ylabel("Testing time in seconds")
    plt.title("Testing time")
    plt.show()

    #Score bar graph
    plt.bar(models, score , color='green', width=0.5)
    plt.xlabel("Model name")
    plt.ylabel("Score")
    plt.title("Score")
    plt.show()


def visualization(x , names , y):

    #Draw bar graph
    plt.bar(x, y , color='green' , width=0.5)
    plt.xticks(x)
    plt.xlabel("News number")
    plt.ylabel("Frequncy")
    plt.title("Frequncy of each news")
    plt.show()

    #Draw bie chart
    plt.pie(y , labels=names)
    plt.show()


def load_model(name):
    model = pickle.load(open(name, 'rb'))
    return model

def test():

    x = np.loadtxt('data_test.txt')
    y = np.loadtxt('target_test.txt')

    logistic_regression = load_model('log_reg_model.pkl')
    print('Logistic regression score: ' , logistic_regression.score(x , y))

    SVM = load_model('svm_model.pkl')
    print('SVM score: ', SVM.score(x, y))

    NB = load_model('naive_bayes_model.pkl')
    print('Naive Bayes score: ', NB.score(x, y))



#Testing
test()

# Visualization of data set
df = fetch_20newsgroups(subset='all')
x = []
y = []

for i in range(0,20):
    x.append(i)
    y.append(0)

for i in df['target']:
    y[i] = y[i] + 1  # frequncy array
visualization(x , df['target_names'] , y)



#Draw bar graphs for the 3 models
training_time = np.loadtxt("training_time.txt")
testing_time = np.loadtxt("testing_time.txt")
score = np.loadtxt("score.txt")
plot_bar_graphs(training_time , testing_time , score)