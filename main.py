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
from pre import pre_process

def fit_logistic_regression(x_train, y_train, x_test, y_test):

    log_reg = LogisticRegression(max_iter=300 , C=10)

    start_time = time.time()
    log_reg.fit(x_train, y_train)
    training_time = time.time() - start_time

    start_time = time.time()
    y_pred = log_reg.predict(x_test)
    testing_time = time.time() - start_time

    pickle.dump(log_reg, open('log_reg_model.pkl', 'wb'))

    print("Training time = ", training_time)
    print("Testing time = ", testing_time)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("score:", log_reg.score(x_test, y_test))

    return [training_time, testing_time, log_reg.score(x_test, y_test)]


def fit_svm_model(x_train, y_train, x_test, y_test):

    svm_model = svm.SVC(max_iter=300 , C=7, kernel='linear')

    start_time = time.time()
    svm_model.fit(x_train, y_train)
    training_time = time.time() - start_time

    start_time = time.time()
    y_pred = svm_model.predict(x_test)
    testing_time = time.time() - start_time

    pickle.dump(svm_model, open('svm_model.pkl', 'wb'))

    print("Training time = ", training_time)
    print("Testing time = ", testing_time)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("score:", svm_model.score(x_test, y_test))

    return [training_time, testing_time, svm_model.score(x_test, y_test)]


def fit_naive_bayes_model(x_train , y_train , x_test , y_test):

    nb_model = GaussianNB()

    start_time = time.time()
    nb_model.fit(x_train, y_train)
    training_time = time.time() - start_time

    start_time = time.time()
    y_pred = nb_model.predict(x_test)
    testing_time = time.time() - start_time

    pickle.dump(nb_model, open('naive_bayes_model.pkl', 'wb'))

    print("Training time = ", training_time)
    print("Testing time = ", testing_time)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("score:", nb_model.score(x_test, y_test))

    return [training_time, testing_time, nb_model.score(x_test, y_test)]



#This part is the part of reading data set, training and saving the models

#Reading data set
df = fetch_20newsgroups(subset='all')
tmp = pre_process(df)

#vectorization
cv = CountVectorizer(max_features = 10000)
x = cv.fit_transform(tmp).toarray() #cv.fit_transform returns pairs, so we must convert it to array
y = df["target"]

x , y = shuffle(x , y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
np.savetxt("data_test.txt" , x_test , fmt='%.5e')
np.savetxt("target_test.txt" , y_test , fmt='%.5e')


train_time_log , test_time_log , score_log = fit_logistic_regression(x_train , y_train , x_test , y_test)
train_time_svm , test_time_svm , score_svm = fit_svm_model(x_train , y_train , x_test , y_test)
train_time_nb , test_time_nb , score_nb = fit_naive_bayes_model(x_train , y_train , x_test , y_test)

training_time = np.array([train_time_log , train_time_svm , train_time_nb])
testing_time = np.array([test_time_log , test_time_svm , test_time_nb])
score = np.array([score_log , score_svm , score_nb])

np.savetxt("training_time.txt" , training_time)
np.savetxt("testing_time.txt" , testing_time)
np.savetxt("score.txt" , score)