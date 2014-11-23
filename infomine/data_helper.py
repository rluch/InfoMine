#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_helper.py
"""
import os


def get_data_file_path(filename):
    """
    Returns the absolute system path of a file residing in
    the data directory for easy fileopening
    """
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    absolute_path = os.path.abspath(os.path.join(data_dir, filename))
    return absolute_path
