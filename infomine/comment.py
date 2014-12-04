#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Module containing methods for comment preprocessing (cleaning) """


class Comment(object):
    """
    Comment Entity.
    Besides getters and setters it handlers simple preprocessing methods
    """
    def __init__(self, comment_string):
        self._comment = comment_string
        self._author = None
        self._gender = None

    @property
    def comment(self):
        return self._comment

    @comment.setter
    def comment(self, value):
        self._comment = value

    @property
    def author(self):
        return self._author

    @author.setter
    def author(self, value):
        self._author = value

    @property
    def male_likes(self):
        return self._male_likes

    @male_likes.setter
    def male_likes(self, value):
        self._male_likes = value

    @property
    def female_likes(self):
        return self._female_likes

    @female_likes.setter
    def female_likes(self, value):
        self._female_likes = value

    @property
    def likes(self):
        """ Returns the calculated sum of male and female likes """
        return sum(self._male_likes, self._female_likes)

    @property
    def like_ratio(self):
        """ Returns the male/female like ratio """
        return 0.0

    def __str__(self):
        return '%s' % self._comment

    def lower_comment(self):
        self._comment = self._comment.lower()
