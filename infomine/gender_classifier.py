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
from sklearn import ensemble
from sklearn import linear_model
import numpy as np
import sklearn
import data_helper
from pylab import *


# class gender_classifier():
#     def tokenize_line(self, comment):
#         return nltk.word_tokenize(comment)

def load_gender_with_comments_from_file(filename):

    training_set_file = data_helper.get_data_file_path(filename+'.csv')

    trainingSet = []

    data_dir = os.path.join(os.path.dirname(__file__), '../data')

    with open(os.path.join(data_dir, training_set_file), 'r') as in_file:
        for line in csv.reader(in_file):
            trainingSet.append((line[1].decode("utf-8"), line[0]))

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

def preprocessing(comment):

    for c in comments:
        words = nltk.word_tokenize(c)

        lower_words = []
    
        for word in words:
            lower_words.append(word.lower())

        comment = lower_word
    return comment


def feature_extractor(comment, sentiment):

    features = {}

    gender_linked_cues_1 = ["yndig".decode("utf-8"), "charmerende".decode("utf-8"), "sød".decode("utf-8"),
                            "dejlig".decode("utf-8"), "guddommelig".decode("utf-8")]
    gender_linked_cues_2 = ["herregud".decode("utf-8"), "hey".decode("utf-8"), "ah".decode("utf-8"), "okay".decode("utf-8")]
    gender_linked_cues_3 = ["wow".decode("utf-8"), "fedt".decode("utf-8"), "cool".decode("utf-8"), "lækkert".decode("utf-8")]
    gender_linked_cues_4 = ["godt".decode("utf-8"), "slags".decode("utf-8"), "en slags".decode("utf-8"), "evt".decode("utf-8"),
                            "måske".decode("utf-8"), "evt.".decode("utf-8"), "muligivs".decode("utf-8")]
    gender_linked_cues_5 = ["virkelig".decode("utf-8"), "meget".decode("utf-8"), "temmelig".decode("utf-8"),
                            "ganske".decode("utf-8"), "særligt".decode("utf-8")]
    gender_linked_cues_6 = ["distraherende".decode("utf-8"), "irriterende".decode("utf-8"), "rar".decode("utf-8")]
    gender_linked_cues_7 = ["undrer".decode("utf-8"), "overveje".decode("utf-8"), "formoder".decode("utf-8"), "antager".decode("utf-8")]

    words = nltk.word_tokenize(comment)
    sentences = nltk.sent_tokenize(comment)
    #part_of_speech = nltk.pos_tag(words)
    #print part_of_speech
    features["number_of_words"] = len(words)
    features["number_of_sentences"] = len(sentences)

    lower_words = []

    for word in words:
        lower_words.append(word.lower())
    features["gender_linked_cues_1"] = len(set(words).intersection(set(gender_linked_cues_1)))
    features["gender_linked_cues_2"] = len(set(words).intersection(set(gender_linked_cues_2)))
    features["gender_linked_cues_3"] = len(set(words).intersection(set(gender_linked_cues_3)))
    features["gender_linked_cues_4"] = len(set(words).intersection(set(gender_linked_cues_4)))
    features["gender_linked_cues_5"] = len(set(words).intersection(set(gender_linked_cues_5)))
    features["gender_linked_cues_6"] = len(set(words).intersection(set(gender_linked_cues_6)))
    features["gender_linked_cues_7"] = len(set(words).intersection(set(gender_linked_cues_7)))
    # check if word appears in the dictionary created ealier.

    sentValue = []
    for word, value in sentiment.iteritems():
        if word in lower_words:
            sentValue.append(int(value))
    if len(sentValue) > 0:
        avg_sent_of_comment = sum(sentValue)/len(sentValue)
        abs_sent_of_comment = max(abs(i) for i in sentValue)
        min_sent_of_comment = min(sentValue)
    else:
        avg_sent_of_comment = 0
        abs_sent_of_comment = 0
        min_sent_of_comment = 0

    features["average_sentiment"] = avg_sent_of_comment
    features["abs_sentiment"] = abs_sent_of_comment
    #features["minimum_sentiment"] = min_sent_of_comment

    ## LEXICAL DIVERSITY FEATURES ##

    return features

def feature_extractor_to_scikitLearn(featureset):

    # sklearn algorithms use numpy arrays and nltk uses dictionary
    # Therefore the conversion
    label = []
    feature_set = []

    for features in featureset:

        now = float(features[0]["number_of_words"])
        nos = float(features[0]["number_of_sentences"])
        avg_sen = float(features[0]["average_sentiment"])
        abs_sen = float(features[0]["abs_sentiment"])
        #min_sen = float(features[0]["minimum_sentiment"])
        glc1 = float(features[0]["gender_linked_cues_1"])
        glc2 = float(features[0]["gender_linked_cues_2"])
        glc3 = float(features[0]["gender_linked_cues_3"])
        glc4 = float(features[0]["gender_linked_cues_4"])
        glc5 = float(features[0]["gender_linked_cues_5"])
        glc6 = float(features[0]["gender_linked_cues_6"])
        glc7 = float(features[0]["gender_linked_cues_7"])

        feature_set.append([now, nos, avg_sen, abs_sen, glc1, glc2, glc3, glc4, glc5, glc6, glc7])
        if features[1] == "Female":
            label.append([1])
        else:
            label.append([0])

    array_list = np.array(feature_set)

    y = np.array(label)

    attributes_names = ["number_of_words", "number_of_sentences", "average_sentiment"]
    class_names = ["Male", "Female"]

    return array_list, y, attributes_names, class_names

def standardize_features(Xfeatures):

    X_scaled = sklearn.preprocessing.scale(Xfeatures)

    return X_scaled

def generate_feature_set(training, sentiment):

    feature_set = [(feature_extractor(comment, sentiment), gender) for (comment, gender) in training]

    return feature_set

def naive_bayes_classification(featuresets):

    splitdata = len(featuresets)/2
    train_set, test_set = featuresets[:splitdata], featuresets[splitdata:]
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    test_accuracy = nltk.classify.accuracy(classifier, test_set)
    train_accuracy = nltk.classify.accuracy(classifier, train_set)
    classifier.show_most_informative_features(10)

    return train_accuracy, test_accuracy

def classification(Xfeatures, Ylabel, algorithm):

    X = Xfeatures
    y = Ylabel
    #print X, y
    N, M = X.shape
    #print type(X), type(y)

    ## Crossvalidation
    # Create crossvalidation partition for evaluation
    K = 10
    #cv = cross_validation.KFold(N, n_folds=K,shuffle=True)
    cv = cross_validation.StratifiedKFold(y.ravel(), n_folds=K, shuffle=True)

    # Initialize variables
    test_accuracy = np.empty((K,1))
    feature_importance = np.empty([K,M])
    k=0
    for train_index, test_index in cv:

        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index,:]
        X_test = X[test_index,:]
        y_test = y[test_index,:]

        # Fit and evaluate Logistic Regression classifier
        if algorithm == "svm":
            clf = svm.SVC()
        elif algorithm == "random_forest":
            clf = ensemble.RandomForestClassifier()
        elif algorithm == "logistic_regression":
            clf = linear_model.LogisticRegression()
        elif algorithm == "ada_boost":
            clf = ensemble.AdaBoostClassifier()

        clf.fit(X_train, y_train.ravel())
        y_est = clf.predict(X_test)
        test_accuracy[k] = sklearn.metrics.accuracy_score(y_test, y_est)

        if algorithm == "random_forest" or "ada_boost":
            feature_importance[k] = clf.feature_importances_

        k+=1

    overall_test_accuracy = sum(test_accuracy)/len(test_accuracy)
    if algorithm == "random_forest" or "ada_boost":
        overall_feature_importance = feature_importance.sum(axis=0)/len(feature_importance)
    else:
        overall_feature_importance = None
    return overall_test_accuracy, overall_feature_importance

def pca_plot(X, y, attributes_names, class_names):

    # Compute values of N, M and C.
    N = len(y)
    M = len(an)
    C = len(cn)

    # Subtract mean value from data
    Y = X - np.ones((N, 1))*X.mean(0)

    # PCA by computing SVD of Y
    U,S,V = linalg.svd(Y,full_matrices=False)
    V = mat(V).T

    # Project the centered data onto principal component space
    Z = Y * V

    # Indices of the principal components to be plotted
    i = 0
    j = 1

    # Plot PCA of the data
    f = figure()
    f.hold()
    title('Information data')
    for c in range(C):
        # select indices belonging to class c:
        class_mask = y.ravel()==c
        plot(Z[class_mask,i], Z[class_mask,j], 'o')
    legend(cn)
    xlabel('PC{0}'.format(i+1))
    ylabel('PC{0}'.format(j+1))

    # Output result to screen
    show()

    return f

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
print training[0]
#print training[0]
sentiment_danish = sentiment_danish_words()
#testing = preprocessing(training[50][0], sentiment_danish)
#print testing

feature_set = generate_feature_set(training[0:100], sentiment_danish)
test, train = naive_bayes_classification(feature_set)
print test, train
X, y, an, cn = feature_extractor_to_scikitLearn(feature_set[0:100])
X_scaled = standardize_features(X)
xtest = classification(X_scaled, y, "random_forest")
print an, cn
print xtest
#print xtest1
#pca_plot(X, y, an, cn)


