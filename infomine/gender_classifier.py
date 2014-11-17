#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
##print nltk.__version__
from collections import Counter
#import DataCollection as dc
import csv

# class gender_classifier():
#     def tokenize_line(self, comment):
#         return nltk.word_tokenize(comment)


def load_gender_with_comments_from_file(filename):

    training_set_file = filename+'.csv'

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
            word.append(line.split('\t')[0]) # Each column in the file is tab seperated
            tab_split = line.split('\t')[1]
            newline_split = tab_split.split('\n')[0]
            sentScore.append(newline_split)

    sentiment = dict(zip(word, sentScore))

    return sentiment

def preprocessing(comment, sentiment):
    features = {}
    words = nltk.word_tokenize(comment.decode("utf-8"))
    sentences = nltk.sent_tokenize(comment.decode("utf-8"))
    features["number_of_words"] = len(words)
    features["number_of_sentences"] = len(sentences)

    lower_words = []
    count = 0
    sentValue = 0
    
    for word in words:
        lower_words.append(word.lower())

    #print list(filter((lambda key: key in lower_words), sentiment))
    # check if word appears in the dictionary created ealier.

    for key in sentiment:
        if key in lower_words:
            count += 1
            sentValue += float(sentiment[key])
    if count > 0:
        avg_sent_of_comment = sentValue/count
    else:
        avg_sent_of_comment = 0
    features["average_sentiment"] = avg_sent_of_comment
    return features


def naive_bayes_classification(features, sentiment):

    splitdata = len(features)/2

    featuresets = [(preprocessing(comment, sentiment), gender) for (comment, gender) in features]
    train_set, test_set = featuresets[:splitdata], featuresets[splitdata:]
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    test_accuracy = nltk.classify.accuracy(classifier, test_set)
    train_accuracy = nltk.classify.accuracy(classifier, train_set)
    classifier.show_most_informative_features(5)
    print test_accuracy
    print train_accuracy

    return featuresets

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
        #print featuresets

training = load_gender_with_comments_from_file("gender_and_comments")
sentiment_danish = sentiment_danish_words()
#testing = preprocessing(training[50][0], sentiment_danish)
#print testing
featureSet = naive_bayes_classification(training[0:10], sentiment_danish)

#print sentiment


#whodat = gender_classifier("""@Torben Nielsen: Vi skal passe på, at vi ikke lader os styre af frygt.
#Da jeg var ung, skulle vi alle sammen være bange for russerne. Det viste sig senere, at de ikke var farligere end andre mennesker. Så skulle vi være bange for AIDS. Vi er adskillige, som har fået vores sexuelle debut uden at bukke under for AIDS.
#"De fremmede", terror og økonomisk og økologisk ruin er senere kommet til.
#I gamle dage var det religion, som var "opium for folket" - senere er der kommet mange andre "stoffer" til.""")