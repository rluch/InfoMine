#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
##print nltk.__version__
from collections import Counter
#import DataCollection as dc
import csv
import os
from sklearn import cross_validation
from sklearn import svm
from sklearn import preprocessing
import numpy as np
import sklearn
import data_helper


# class gender_classifier():
#     def tokenize_line(self, comment):
#         return nltk.word_tokenize(comment)

def load_gender_with_comments_from_file(filename):

    training_set_file = data_helper.get_data_file_path(filename+'.csv')

    trainingSet = []

    with open(training_set_file, "r") as in_file:
        for line in csv.reader(in_file):
            trainingSet.append((line[1], line[0]))

    return trainingSet

def sentiment_danish_words():
    word = []
    sentScore = []
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    
    # Pair each wood with its average sentiment
    with open(os.path.join(data_dir, 'Nielsen2011Sentiment_afinndk-2.txt'), 'r') as in_file:
        for line in in_file.readlines()[0:]:
            word.append(line.split('\t')[0].decode("utf-8"))  # Each column in the file is tab seperated
            tab_split = line.split('\t')[1]
            newline_split = tab_split.split('\n')[0]
            sentScore.append(newline_split)

    sentiment = dict(zip(word, sentScore))

    return sentiment

def preprocessing(comments):

    for c in comments:
        words = nltk.word_tokenize(c.decode("utf-8"))

        lower_words = []
    
        for word in words:
            lower_words.append(word.lower())

        comment = lower_word
    return comment


def feature_extractor(comment, sentiment):

    features = {}
    words = nltk.word_tokenize(comment.decode("utf-8"))
    sentences = nltk.sent_tokenize(comment.decode("utf-8"))
    features["number_of_words"] = len(words)
    features["number_of_sentences"] = len(sentences)

    lower_words = []

    for word in words:
        lower_words.append(word.lower())

    # check if word appears in the dictionary created ealier.
    count = 0
    sentValue = 0

    for key, value in sentiment.iteritems():
        if key in lower_words:
            count += 1
            sentValue += int(value)
    if count > 0:
        avg_sent_of_comment = sentValue/count
    else:
        avg_sent_of_comment = 0
    features["average_sentiment"] = avg_sent_of_comment
    return features

def feature_extractor_to_scikitLearn(featureset):

    number_of_words = []
    number_of_sent = []
    sentiment = []
    label = []

    for features in featureset:
        number_of_words.append(float(features[0]["number_of_words"]))
        number_of_sent.append(float(features[0]["number_of_sentences"]))
        sentiment.append(float(features[0]["average_sentiment"]))
        if features[1] == "Female":
            label.append([1])
        else:
            label.append([0])

    arraylist = np.array([number_of_words] + [number_of_sent] + [sentiment])
    Xfeatures = arraylist.transpose()
    Ylabel = np.array(label)
    #print Ylabel

    #Standardize features
    X_scaled = sklearn.preprocessing.scale(Xfeatures)

    attributes_names = ["number_of_words", "number_of_sentences", "average_sentiment" ]
    class_names = ["Female", "Male"]
    #print Ylabel
    #print X_scaled
    return X_scaled, Ylabel

def generate_feature_set(training, sentiment):

    feature_set = [(feature_extractor(comment, sentiment), gender) for (comment, gender) in training]

    return feature_set


def naive_bayes_classification(featuresets):

    splitdata = len(featuresets)/2
    train_set, test_set = featuresets[:splitdata], featuresets[splitdata:]
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    test_accuracy = nltk.classify.accuracy(classifier, test_set)
    train_accuracy = nltk.classify.accuracy(classifier, train_set)
    classifier.show_most_informative_features(5)

    return train_accuracy, test_accuracy

def svm_classification(Xfeatures, Ylabel):

    X = Xfeatures
    y = Ylabel

    #print X, y
    N, M = X.shape
    #print type(X), type(y)

    ## Crossvalidation
    # Create crossvalidation partition for evaluation
    K = 4
    #cv = cross_validation.KFold(N, n_folds=K,shuffle=True)
    cv = cross_validation.StratifiedKFold(y.ravel(), n_folds=K, shuffle=True)

    # Initialize variables
    test_accuracy = np.empty((K,1))

    k=0
    for train_index, test_index in cv:

        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index,:]
        X_test = X[test_index,:]
        y_test = y[test_index,:]

        # Fit and evaluate Logistic Regression classifier
        clf = svm.SVC()
        clf.fit(X_train, y_train.ravel())
        y_est = clf.predict(X_test)
        test_accuracy[k] = sklearn.metrics.accuracy_score(y_test, y_est)

        #y_logreg = np.mat(model.predict(X_test)).T
        #Error_logreg[k] = 100*(y_est!=y_test).sum().astype(float)/len(y_test)

        k+=1

    #splitdata = len(X)/2

    #X_train, X_test = X[splitdata:], X[:splitdata]
    #y_train, y_test = y[splitdata:], y[:splitdata]

    #clf = svm.SVC()
    #clf.fit(X_train, y_train)
    #y_est = clf.predict(X_test)


    #test_accuracy = sklearn.metrics.accuracy_score(y_test, y_est)
    #train_accuracy = sklearn.metrics.accuracy_score(y_train, y_est)

    return test_accuracy
    #skf = cross_validation.StratifiedKFold(y, n_folds=2)

    #print skf

    #for train_index, test_index in skf:
    #    print("TRAIN:", train_index, "TEST:", test_index)
    #    X_train, X_test = X[train_index], X[test_index]
    #    y_train, y_test = y[train_index], y[test_index]

    #return X_train, X_test

### Rasmus ###
 #def feature_extractor(self, comment):
    # features = {}
    # words = nltk.word_tokenize(comment[0])
    # sentences = nltk.sent_tokenize(comment[0])
    # print words
    # features["number_of_words"] = len(words)
    # features["number_of_sentences"] = len(sentences)
    # #tokenized = self.tokenize_line(comment.translate(None, string.punctuation))
    # #tokenized = self.tokenize_line(comment)
    # #wordpunct_tokenize(raw)
    # #for word in tokenized:
    # #    if word not in nltk.corpus.stopwords.words('danish') and word not in '.,-@#"\'':
    # #        if word not in features:
    # #            features.append(word)
    # return features


    #def __init__(self, comment):
        #features = self.preprocessing(comment)
        #featuresets = self.naive_bayes_classification(comment)
        #featuresets = self.naive_bayes_classification(comment)
        #print featuresets

training = load_gender_with_comments_from_file("gender_and_comments")
#print training[0:100]
sentiment_danish = sentiment_danish_words()
#testing = preprocessing(training[50][0], sentiment_danish)
#print testing
feature_set = generate_feature_set(training[0:600], sentiment_danish)
x, y = feature_extractor_to_scikitLearn(feature_set[0:600])
xtest = svm_classification(x, y)

print xtest
#print feature_set[0][0]
#classifier = naive_bayes_classification(feature_set)

#comment = preprocessing(training[0:10])
#print comment
#print sentiment


#whodat = gender_classifier("""@Torben Nielsen: Vi skal passe på, at vi ikke lader os styre af frygt.
#Da jeg var ung, skulle vi alle sammen være bange for russerne. Det viste sig senere, at de ikke var farligere end andre mennesker. Så skulle vi være bange for AIDS. Vi er adskillige, som har fået vores sexuelle debut uden at bukke under for AIDS.
#"De fremmede", terror og økonomisk og økologisk ruin er senere kommet til.
#I gamle dage var det religion, som var "opium for folket" - senere er der kommet mange andre "stoffer" til.""")