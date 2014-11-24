#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Module containing methods for comment preprocessing (cleaning) """


class Comment(object):
    def __init__(self, comment_string):
        self._comment = comment_string

    @property
    def comment(self):
        return self._comment

    @comment.setter
    def comment(self, value):
        self._comment = value

    def prep_lower(self):
        self.comment = self._comment.lower()
