#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_helper.py
"""
import os
import sys
import csv
import pickle
import random


def get_data_file_path(filename):
    """
    Returns the absolute system path of a file residing in
    the data directory for easy fileopening
    """
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    absolute_path = os.path.abspath(os.path.join(data_dir, filename))
    return absolute_path


def load_and_return_lines_from_csv_file(filename):
    """
    Generic methos for loading and returning lines of any file
    """
    data_file = get_data_file_path(filename)
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    lines = []
    with open(os.path.join(data_dir, data_file), 'r') as in_file:
        for line in csv.reader(in_file):
            lines.append(line)
    return lines


def load_serialized_comments_from_file(filename):
    filepath = get_data_file_path(filename)
    comments = []
    try:
        comments = pickle.load(open(filepath, 'rb'))
    except IOError, e:
        print "[ERROR] Comments could not loaded!"
        print "\tRun infominer with \"--mysql\" to extract from mysql.\n"
        sys.exit(e)
    return comments


def gender_ratio_normalize_comments(_comments):
    """
    Prepare for training. Discard unknown gender and
    ensure the same amount of male and female samples
    """
    comments = sorted(_comments, key=lambda comment: comment.gender)
    comments_male = []
    comments_female = []
    for c in comments:
        if c.gender == "male":
            comments_male.append(c)
        elif c.gender == "female":
            comments_female.append(c)
    male_diff = len(comments_male) - len(comments_female)
    random.shuffle(comments_male)
    comments_male = comments_male[:len(comments_female)]

    comments = comments_male + comments_female
    random.shuffle(comments)
    return comments


def load_afinndk_sentiment_file_to_dict():
    filepath = get_data_file_path('Nielsen2011Sentiment_afinndk-2.txt')
    word = []
    sentScore = []

    with open(filepath, 'r') as in_file:
        for line in in_file.readlines()[0:]:
            # Each column in the file is tab seperated
            word.append(line.split('\t')[0].decode("utf-8"))
            tab_split = line.split('\t')[1]
            newline_split = tab_split.split('\n')[0]
            sentScore.append(newline_split)

    # Pair each word with its average sentiment and return it
    sentiment = dict(zip(word, sentScore))
    return sentiment
