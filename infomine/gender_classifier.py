#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
##print nltk.__version__
from collections import Counter
from DataCollection import DataCollection

class gender_classifier():
    def tokenize_line(self, comment):
        return nltk.word_tokenize(comment)

    def sentiment_danish_words(self):
        word = []
        sentScore = []

        # Pair each wood with its average sentiment
        with open("/Users/Henrik/InfoMine/data/Nielsen2011Sentiment_afinndk-2.txt", 'r') as in_file:
            for line in in_file.readlines()[0:]:
                word.append(line.split('\t')[0]) # Each column in the file is tab seperated
                tab_split= line.split('\t')[1]
                newline_split = tab_split.split('\n')[0]
                sentScore.append(newline_split)

        sentiment = dict(zip(word, sentScore))

        return sentiment

    def preprocessing(self, comment):
        features = {}
        words = nltk.word_tokenize(comment.decode("utf-8"))
        sentences = nltk.sent_tokenize(comment.decode("utf-8"))
        features["number_of_words"] = len(words)
        features["number_of_sentences"] = len(sentences)

        return features

    def naive_bayes_classification(self, features):
        splitdata = len(features)/2

        featuresets = [(self.preprocessing(comment), gender) for (comment, gender) in features]
        train_set, test_set = featuresets[:splitdata], featuresets[splitdata:]
        classifier = nltk.NaiveBayesClassifier.train(train_set)

        testError = nltk.classify.accuracy(classifier, test_set)
        trainError = nltk.classify.accuracy(classifier, train_set)
        classifier.show_most_informative_features(3)
        print testError
        print trainError

        return featuresets

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

training = DataCollection().load_gender_with_comments_from_file("gender_and_comments")
#testing = gender_classifier().preprocessing(training[50][0])
#featureSet = gender_classifier().naive_bayes_classification(training)
sentiment = gender_classifier().sentiment_danish_words()
print sentiment

#whodat = gender_classifier("""@Torben Nielsen: Vi skal passe på, at vi ikke lader os styre af frygt.
#Da jeg var ung, skulle vi alle sammen være bange for russerne. Det viste sig senere, at de ikke var farligere end andre mennesker. Så skulle vi være bange for AIDS. Vi er adskillige, som har fået vores sexuelle debut uden at bukke under for AIDS.
#"De fremmede", terror og økonomisk og økologisk ruin er senere kommet til.
#I gamle dage var det religion, som var "opium for folket" - senere er der kommet mange andre "stoffer" til.""")