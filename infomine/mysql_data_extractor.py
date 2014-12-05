# -*- coding: utf-8 -*-
import mysql.connector
import pickle
from comment import Comment
from gender_helper import GenderHelper
from data_helper import get_data_file_path


class MySQLDataExtractor(object):
    def __init__(self):
        self.firstnames_and_ids = {}
        self.comments = []
        self.gh = GenderHelper()

    def establish_new_mysql_connection(self):
        return mysql.connector.connect(
            user='root',
            password='',
            host='127.0.0.1',
            database='Information')

    def save_comments_to_file(self, filename):
        """
        Saving/overwrite extracted comment objects serialized
        in a pickle-file for easy loading.
        """
        path = get_data_file_path(filename)
        pickle.dump(
            self.comments,
            open(path, 'wb'))
        print("[INFO] Saved %d comments to %s" % (len(self.comments), path))

    def load_firstnames_from_mysql(self):
        """
        Function to load in commenters first name from SQL dump
        """
        cnx = self.establish_new_mysql_connection()
        cursor = cnx.cursor()
        query = ("SELECT u_name, u_uid FROM Information.inf_dtu_user")
        cursor.execute(query)
        for c in cursor:
            if c[0] is not None:
                ckey = int(str(c[1]).decode("utf-8"))
                cval = c[0].split()[0].lower()
                self.firstnames_and_ids[ckey] = cval
        cursor.close()
        cnx.close()

    def load_comments_and_gender_and_comment_likes_from_mysql(self):
        """
        Function to load in comments, gender and comment likes from SQL dump
        Returns a list that contain gender of the commenter, comment body,
        male comment likes, female comment likes, total likes and male/female ratio comment like
        """
        cnx = self.establish_new_mysql_connection()
        cursor = cnx.cursor()
        query = ("""
            SELECT u.u_name, c.c_body, group_concat(cf.f_uid)
            As like_ids FROM Information.inf_dtu_user u,
            Information.inf_dtu_comment c,
            Information.inf_dtu_comment_flag cf where u.u_uid=c.c_uid
            and c.c_cid = cf.f_cid group by c.c_cid;
            """)
        cursor.execute(query)

        # Makes a list that for each row contains name,
        # comments and name of the persons who liked the comment
        for c in cursor:
            if c[0] is not None:
                _male_likes = 0
                _female_likes = 0
                comment = Comment(c[1])
                comment.author = c[0].split()[0].lower()
                # Determines if the commenter is a male or female
                comment.gender = self.gh.get_gender_by_name(comment.author)
                like_ids = c[2].split(",")
                # Determines what gender the comment likes are
                # and count the number of male, female and total comment likes
                for like_id in like_ids:
                    if int(like_id) in self.firstnames_and_ids.keys():
                        lname = self.firstnames_and_ids[int(like_id)]
                        gender = self.gh.get_gender_by_name(lname)
                        if gender == "male":
                            _male_likes += 1
                        elif gender == "female":
                            _female_likes += 1
                comment.male_likes = _male_likes
                comment.female_likes = _female_likes
                self.comments.append(comment)
