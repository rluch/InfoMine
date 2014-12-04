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

    def __str__(self):
        return '%s' % self._comment

    def lower_comment(self):
        self._comment = self._comment.lower()
