# -*- coding: utf-8 -*-
"""
Author gender identification with machine learning.
"""
__version__ = '0.1.0'
__author__ = 'Henrik Holm, Rasmus Lundsgaard'
__author_email__ = 's103214@student.dtu.dk, s123344@student.dtu.dk'

from comment import Comment
from data_helper import load_and_return_lines_from_csv_file
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

        if(args['<comment>'] i not False):
            c = Comment(args['<comment>'])
            print '[OPT] Identifying one comment.'
        elif(args['<comments_file>']):
            comments = []
            raw_comments = load_and_return_lines_from_csv_file("testingNew.csv")
            for lines in raw_comments:
                c = Comment(lines[1])
                comments.append(c)
            print "comments file %s" % len(comments)

        print "[*] Loading comments from data/comments.p ..."
        comments = load_serialized_comments_from_file('comments.p')
        print("[INFO] Comments loaded from file: %d" % len(comments))
        #raw_comments = load_and_return_lines_from_csv_file("testingNew.csv")
        print "[*] Training classifier with loaded training data ..."
