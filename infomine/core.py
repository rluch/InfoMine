# -*- coding: utf-8 -*-
"""
Author gender identification with machine learning.
"""
__version__ = '0.1.0'
__author__ = 'Henrik Holm, Rasmus Lundsgaard'
__author_email__ = 's103214@student.dtu.dk, s123344@student.dtu.dk'

from comment import Comment


class InfoMiner(object):
    """InfoMiner core launcher module"""
    def __init__(_class, args):
        """
        InfoMiner Constructor.
        Parses and acts on supplied docopts
        """
        print(_class.parse_opts(args))

    def parse_opts(self, args):
        """
        Acts on parsed CLI parameters.
        """
        if(args['<comment>']):
            print(args['<comment>'])
            return Comment(args['<comment>'])
        elif(args['<comments_file>']):
            print "comments file"
