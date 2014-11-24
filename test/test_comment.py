#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Module containing methods for comment preprocessing (cleaning) """
from infomine.comment import Comment


def test_prep_lower():
    test_comment = 'Tror ikke hr. Lars Dahl forstår sarkasmen ...'
    test_comment_expected = 'tror ikke hr. lars dahl forstår sarkasmen ...'
    c = Comment(test_comment)
    c.prep_lower()
    actual = c.comment
    assert actual == test_comment_expected
