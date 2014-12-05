#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
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
from nltk.corpus import stopwords
import numpy as np
import sklearn
import data_helper
#from pylab import *

import time

from nltk.util import tokenwrap
from data_helper import load_afinndk_sentiment_file_to_dict, load_serialized_comments_from_file


def load_gender_with_comments_from_file(filename):

    data_set_file = data_helper.get_data_file_path(filename+'.csv')

    data_set = []

    data_dir = os.path.join(os.path.dirname(__file__), '../data')

    with open(os.path.join(data_dir, data_set_file), 'r') as in_file:
        for line in csv.reader(in_file):
            data_set.append((line[0].decode("utf-8"), line[1].decode("utf-8"), line[2], line[3], line[4], line[5]))

    return data_set


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

    words = nltk.word_tokenize(comment)
    clean_words = [word.lower() for word in words if word.lower() not in stopwords.words('danish')]
    cleaned_comment = tokenwrap(clean_words)

    return cleaned_comment


def clean_comments(data_set):

    cleaned_data_set = []
    for ds in data_set:
        #clean_comment = preprocessing(ds[1])
        ds.comment = preprocessing(ds.comment)
        cleaned_data_set.append((ds))
        #cleaned_data_set.append((ds[0], clean_comment, ds[2], ds[3], ds[4], ds[5]))
    return cleaned_data_set


def feature_extractor(comment, male_likes, female_likes, total_likes, male_likes_ratio, sentiment):

    features = {}

    words = nltk.word_tokenize(comment)
    sentences = nltk.sent_tokenize(comment)

    ## Kendeord / Article Words
    article_words = ["den".decode("utf-8"), "det".decode("utf-8") , "de".decode("utf-8"), "en".decode("utf-8"), "et".decode("utf-8")]

    ## Pro sentence words
    pro_sentence_words = ["ja".decode("utf-8"), "nej".decode("utf-8"), "ok".decode("utf-8"), "okay".decode("utf-8"), "jamen".decode("utf-8")]

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


    ## Word, character and structural features
    features["number_of_words"] = len(words)
    features["number_of_sentences"] = len(sentences)
    features["lexical_diversity"] = len(comment) / len(set(comment))
    features["number_of_characters"] = len(comment)
    features["average_length_per_word"] = len(comment)/len(words)
    features["average_length_per_sentence"] = len(comment)/len(sentences)

    ## Function words
    features["article_words"] = len(set(words).intersection(set(article_words)))
    features["pro_sentence_words"] = len(set(words).intersection(set(pro_sentence_words)))

    ## Gender features
    features["gender_linked_cues_1"] = len(set(words).intersection(set(gender_linked_cues_1)))
    features["gender_linked_cues_2"] = len(set(words).intersection(set(gender_linked_cues_2)))
    features["gender_linked_cues_3"] = len(set(words).intersection(set(gender_linked_cues_3)))
    features["gender_linked_cues_4"] = len(set(words).intersection(set(gender_linked_cues_4)))
    features["gender_linked_cues_5"] = len(set(words).intersection(set(gender_linked_cues_5)))
    features["gender_linked_cues_6"] = len(set(words).intersection(set(gender_linked_cues_6)))
    features["gender_linked_cues_7"] = len(set(words).intersection(set(gender_linked_cues_7)))

    # check if word appears in the dictionary created earlier.

    sentValue = []
    for word, value in sentiment.iteritems():
        if word in comment:
            sentValue.append(int(value))
    if len(sentValue) > 0:
        avg_sent_of_comment = sum(sentValue)/len(sentValue)
        min_sent_of_comment = min(sentValue)
        max_sent_of_comment = max(sentValue)
    else:
        avg_sent_of_comment = 0
        max_sent_of_comment = 0
        min_sent_of_comment = 0

    ## Sentiment features
    features["average_sentiment"] = avg_sent_of_comment
    features["minimum_sentiment"] = min_sent_of_comment
    features["maximum_sentiment"] = max_sent_of_comment

    ## Network / like features
    features["male_likes"] = male_likes
    features["female_likes"] = female_likes
    features["total_likes"] = total_likes
    features["male_female_likes_ratio"] = male_likes_ratio

    return features


def generate_feature_set(data_set, sentiment):

    feature_set = [(feature_extractor(comment.comment, comment.male_likes, comment.female_likes, comment.likes, comment.likes_ratio, sentiment),
                    comment.gender) for comment in data_set]

    return feature_set


def feature_extractor_to_scikitLearn(featureset):

    # sklearn algorithms use numpy arrays and nltk uses dictionary
    # Therefore the conversion

    label = []
    feature_set = []

    for features in featureset:

        now = float(features[0]["number_of_words"])
        nos = float(features[0]["number_of_sentences"])
        noc = float(features[0]["number_of_characters"])
        alw = float(features[0]["average_length_per_word"])
        als = float(features[0]["average_length_per_sentence"])
        aw = float(features[0]["article_words"])
        psw = float(features[0]["pro_sentence_words"])
        glc1 = float(features[0]["gender_linked_cues_1"])
        glc2 = float(features[0]["gender_linked_cues_2"])
        glc3 = float(features[0]["gender_linked_cues_3"])
        glc4 = float(features[0]["gender_linked_cues_4"])
        glc5 = float(features[0]["gender_linked_cues_5"])
        glc6 = float(features[0]["gender_linked_cues_6"])
        glc7 = float(features[0]["gender_linked_cues_7"])
        lex_div = float(features[0]["lexical_diversity"])
        avg_sen = float(features[0]["average_sentiment"])
        max_sen = float(features[0]["maximum_sentiment"])
        min_sen = float(features[0]["minimum_sentiment"])
        ml = float(features[0]["male_likes"])
        fl = float(features[0]["female_likes"])
        tl = float(features[0]["total_likes"])
        mlcfl = float(features[0]["male_female_likes_ratio"])

        feature_set.append([now, nos, noc, alw, als, aw, psw, glc1, glc2, glc3, glc4, glc5, glc6, glc7, lex_div,
                            avg_sen, max_sen, min_sen, ml, fl, tl, mlcfl])

        if features[1] == "female":
            label.append([1])
        else:
            label.append([0])

    array_list = np.array(feature_set)

    y = np.array(label)

    attributes_names = ["number_of_words", "number_of_sentences", "number_of_characters", "average length per word",
                        "average length per sentence", "article_words", "pro_sentence_words", "gender_linked_cues_1",
                        "gender_linked_cues_2", "gender_linked_cues_3", "gender_linked_cues_4", "gender_linked_cues_5",
                        "gender_linked_cues_6", "gender_linked_cues_7", "lexical_diversity", "average_sentiment",
                        "maximum_sentiment", "minimum_sentiment", "male_likes", "female_likes", "total_likes",
                        "male_female_likes_ratio"]

    class_names = ["Male", "Female"]

    return array_list, y, attributes_names, class_names


def standardize_features(Xfeatures):

    X_scaled = sklearn.preprocessing.scale(Xfeatures)

    return X_scaled


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
    N, M = X.shape

    # Crossvalidation
    K = 5
    cv = cross_validation.StratifiedKFold(y.ravel(), n_folds=K, shuffle=True)

    # Initialize variables
    test_accuracy = np.empty((K,1))
    train_accuracy = np.empty((K,1))
    feature_importance = np.empty([K,M])

    k = 0
    for train_index, test_index in cv:

        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index,:]
        X_test = X[test_index,:]
        y_test = y[test_index,:]

        # Fit the different classifiers
        if algorithm == "svm":
            clf = svm.SVC()
        elif algorithm == "random_forest":
            clf = ensemble.RandomForestClassifier()
        elif algorithm == "logistic_regression":
            clf = linear_model.LogisticRegression()
        elif algorithm == "ada_boost":
            clf = ensemble.AdaBoostClassifier()

        # evaluate the different classifiers
        clf.fit(X_train, y_train.ravel())
        y_est = clf.predict(X_test)
        test_accuracy[k] = sklearn.metrics.accuracy_score(y_test, y_est)
        train_accuracy[k] = clf.score(X_train, y_train.ravel())

        # Extract feature importances if possible
        if algorithm in ["random_forest", "ada_boost"]:
            feature_importance[k] = clf.feature_importances_
        k += 1

    # Confusion Matrix
    cm = sklearn.metrics.confusion_matrix(y_test, y_est)

    if algorithm == "random_forest" or "ada_boost":
        overall_feature_importance = feature_importance.sum(axis=0)/len(feature_importance)
    else:
        overall_feature_importance = None

    return train_accuracy, test_accuracy, overall_feature_importance, cm, clf


data_set = load_gender_with_comments_from_file("ModifiedDataSet")
comments = load_serialized_comments_from_file('comments.p')
comments = data_helper.gender_ratio_normalize_comments(comments)
print len(comments)

sentiment_danish = sentiment_danish_words()
sentiment_danish = load_afinndk_sentiment_file_to_dict()

#cleaned_data_set = clean_comments(data_set[0:100])
comments = clean_comments(comments)

#feature_set = generate_feature_set(cleaned_data_set, sentiment_danish)
feature_set = 0
feature_set = generate_feature_set(comments, sentiment_danish)
print feature_set[0]

X, y, an, cn = feature_extractor_to_scikitLearn(feature_set)
print X[0]
print y
X = standardize_features(X)

trainAcAB, testAcAB, featureImportanceAB, cmAB, model = classification(X, y, "ada_boost")
trainAcRD, testAcRF, featureImportanceRF, cmRF, model = classification(X, y, "random_forest")

trainAcSVM, testAcSVM, featureImportance, cmSVM, model = classification(X, y, "svm")
trainAcLR, testAcLR, featureImportance, cmLR, model = classification(X, y, "logistic_regression")

print an, cn
print trainAcAB, testAcAB, featureImportanceAB, cmAB
#pca_plot(X, y, an, cn)


print model.predict(X[0])

