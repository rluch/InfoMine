#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
from nltk import word_tokenize

""" Module containing all feature extration methods """


def extract_exclamation_mark_per_word(comment):
    """
    Calculates the amount of exclamation marks per word
    and returns the amount as percentage
    """
    words = word_tokenize(comment.decode("utf-8"))

    pass


counter = Counter(['A', 'A', 'B',
'A', 'C',