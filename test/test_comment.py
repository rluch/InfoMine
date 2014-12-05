#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Module containing methods for comment preprocessing (cleaning) """
from infomine.comment import Comment


def test_comment_init():
    test_value = "Test comment"
    c = Comment(test_value)
    assert c.comment == test_value


def test_comment_comment():
    test_value_1 = "Bad Test Comment"
    test_value_2 = "Good Test Comment"
    c = Comment(test_value_1)
    c.comment = test_value_2
    assert c.comment is not test_value_1
    assert c.comment == test_value_2


def test_comment_lowering():
    test_value = "Good Test Comment"
    c = Comment(test_value)
    c.lower_comment()
    assert c.comment == test_value.lower()
