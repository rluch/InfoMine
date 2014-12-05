#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_helper.py
"""
import os
import sys
import csv
import pickle


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
