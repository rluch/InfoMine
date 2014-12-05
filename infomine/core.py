# -*- coding: utf-8 -*-
"""
Author gender identification with machine learning.
"""
__version__ = '1.0.0'
__author__ = 'Henrik Holm, Rasmus Lundsgaard'
__author_email__ = 's103214@student.dtu.dk, s123344@student.dtu.dk'
import sys
from comment import Comment
from data_helper import load_serialized_comments_from_file
from gender_classifier import train_classifier, classify


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
            sys.exit(0)
        if(args['<comment>'] is not None):
            c = Comment(args['<comment>'])
            print '[OPT] Identifying one comment:'
            print "[*] Loading comments from data/comments.p ..."
            comments = load_serialized_comments_from_file('comments.p')
            print("[INFO] Comments loaded from file: %d" % len(comments))
            print "[*] Training classifier with loaded training data ...\n"
            model, sentiment = train_classifier()
            if classify(c, model[0], sentiment) == 1:
                print "Adaptive Boost identifies as female"
            else:
                print "Adaptive Boost identifies as male"
            if classify(c, model[1], sentiment) == 1:
                print "Random Forest identifies as female"
            else:
                print "Random Forest identifies as male"
            if classify(c, model[2], sentiment) == 1:
                print "SVM identifies as female"
            else:
                print "SVM identifies as male"
            if classify(c, model[3], sentiment) == 1:
                print "Logistic Regression identifies as female"
            else:
                print "Logistic Regression identifies as male"
        else:
            print """
Please supply a comment you want identified!
See \"infominer -h\" for help
            """
