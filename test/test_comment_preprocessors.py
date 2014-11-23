#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Module containing methods for comment preprocessing (cleaning) """

from infomine.gender_classifier import load_gender_with_comments_from_file


def test_citation_preprocessor():
    load_gender_with_comments_from_file("gender_and_comments")
    assert 1 == 1
