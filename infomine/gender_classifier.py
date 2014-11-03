#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk

class gender_classifier():
    def tokenize_line(self, line):
        return nltk.word_tokenize(line)

    def feature_extractor(self, line):
        features = list()
        tokenized = nltk.wordpunct_tokenize(line)
        #tokenized = self.tokenize_line(line.translate(None, string.punctuation))
        #tokenized = self.tokenize_line(line)
        #wordpunct_tokenize(raw)
        for word in tokenized:
            if word not in nltk.corpus.stopwords.words('danish') and word not in '.,-@#"\'':
                if word not in features:
                    features.append(word)
        return features

    def __init__(self, line):
        features = self.feature_extractor(line)
        print features




whodat = gender_classifier("""@Torben Nielsen: Vi skal passe på, at vi ikke lader os styre af frygt.
Da jeg var ung, skulle vi alle sammen være bange for russerne. Det viste sig senere, at de ikke var farligere end andre mennesker. Så skulle vi være bange for AIDS. Vi er adskillige, som har fået vores sexuelle debut uden at bukke under for AIDS.
"De fremmede", terror og økonomisk og økologisk ruin er senere kommet til.
I gamle dage var det religion, som var "opium for folket" - senere er der kommet mange andre "stoffer" til.""")