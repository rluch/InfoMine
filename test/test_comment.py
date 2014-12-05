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


def test_comment_gender():
    test_value = "Test comment"
    test_gender = "unicorn"
    c = Comment(test_value)
    c.gender = test_gender
    assert c.gender == test_gender


def test_comment_female_likes():
    test_value = "Test comment"
    test_likes = 1337
    c = Comment(test_value)
    c.female_likes = test_likes
    assert c.female_likes == test_likes
    assert c.likes == test_likes


def test_comment_male_likes():
    test_value = "Test comment"
    test_likes = 1337
    c = Comment(test_value)
    c.male_likes = test_likes
    assert c.male_likes == test_likes
    assert c.likes == test_likes


def test_comment_likes():
    test_value = "Test comment"
    test_likes_1 = 1337
    test_likes_2 = 1338
    c = Comment(test_value)
    c.male_likes = test_likes_1
    c.female_likes = test_likes_2
    likes = test_likes_1 + test_likes_2
    assert c.likes == likes


def test_comment_likes_ratio():
    test_value = "Test comment"
    test_likes_1 = 1337
    test_likes_2 = 1337
    c = Comment(test_value)
    c.male_likes = test_likes_1
    c.female_likes = test_likes_2
    likes = test_likes_1 + test_likes_2
    ratio = float(test_likes_1) / float(likes)
    assert c.likes_ratio == ratio


def test_comment_lowering():
    test_value = "Good Test Comment"
    c = Comment(test_value)
    c.lower_comment()
    assert c.comment == test_value.lower()
