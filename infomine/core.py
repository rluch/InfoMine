# -*- coding: utf-8 -*-
"""
Author gender identification with machine learning.
"""
__version__ = '0.1.0'
__author__ = 'Henrik Holm, Rasmus Lundsgaard'
__author_email__ = 's103214@student.dtu.dk, s123344@student.dtu.dk'

from comment import Comment
from data_helper import load_serialized_comments_from_file


class InfoMiner(object):
    """InfoMiner core launcher module"""
    def __init__(_class, args):
        """
        InfoMiner Constructor.
        Parses and acts on supplied docopts
        """
        print "Welcome to InfoMiner!\n"
        _class.parse_opts(args)

    def parse_opts(self, args):
        """
        Acts on parsed CLI parameters.
        """
        if(args['--mysql'] is not False):
            print "[OPT] Loading data from local MySQL server ..."
            from mysql_data_extractor import MySQLDataExtractor
            m = MySQLDataExtractor()
            m.load_firstnames_from_mysql()
            m.load_comments_and_gender_and_comment_likes_from_mysql()
            print "[OPT] Saving loaded comments to data/comments.p ..."
            m.save_comments_to_file('comments.p')
            print """Data has been extracted from MySQL.
            Please run the application again!"""
        if(args['<comment>'] is not None):
            c = Comment(args['<comment>'])
            print '[OPT] Identifying one comment:'
            print 'Comment says:\n\t%s' % c.comment
            print "[*] Loading comments from data/comments.p ..."
            comments = load_serialized_comments_from_file('comments.p')
            print("[INFO] Comments loaded from file: %d" % len(comments))
            print "[*] Training classifier with loaded training data ..."
        else:
            print """Please supply a comment you want identified!\n
            See \"infominer -h\" for help"""
