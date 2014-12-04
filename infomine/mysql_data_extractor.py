# -*- coding: utf-8 -*-
import mysql.connector

class MySQLDataExtractor(object):
    def __init__(self):
        self.conn = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1',
                              database='Information')        

    def establish_new_mysql_connection(self):
        return mysql.connector.connect(user='root', password='',
                              host='127.0.0.1',
                              database='Information')
    def save_as_json(self):
        """ Saving extracted data as a JSON-file for future easy parsing """

    def load_firstnames_from_mysql(self):

        user_inf = []

        cnx = self.establish_new_mysql_connection()

        cursor = cnx.cursor()
        query = ("SELECT u_name, u_uid FROM Information.inf_dtu_user")
        cursor.execute(query)

        for c in cursor:
            if c[0] != None:
                name = c[0]
                user_inf.append([name.split()[0].lower(), str(c[1]).decode("utf-8")])

        cursor.close()
        cnx.close()

        return user_inf


    def load_comments_and_gender_and_comment_likes_from_mysql(self, user_inf, gender_name_list):

        cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1',
                              database='Information')

        cursor = cnx.cursor()
        query = ("SELECT u.u_name, c.c_body, group_concat(cf.f_uid) As like_ids FROM Information.inf_dtu_user u, Information.inf_dtu_comment c,"
                 "Information.inf_dtu_comment_flag cf where u.u_uid=c.c_uid and c.c_cid = f_cid group by c.c_cid;")
        cursor.execute(query)

        # Makes a list that for each row contains name, comments and name of the persons who liked the comment
        like_names_comments = []
        for c in cursor:
            if c[0] != None:
                name = c[0]
                firstName = name.split()[0].lower()
                comment = c[1]
                like_ids = c[2].split(",")
                like_names = []
                for ui in user_inf:
                    if ui[1] in like_ids:
                        like_names.append(ui[0])

                like_names_comments.append([firstName, comment, like_names])

        data_set = []
        data_set.append(("Gender", "Comment", "Number_of_male_likes", "Number_of_female_likes", "Total_likes",
                         "Male_likes_compared_to_Female"))

        # Determines if the commenter is a male or female and determines what gender the comment likes are
        # Count the number of male, female and total comment likes
        for lnc in like_names_comments:
            male_count = 0
            female_count = 0
            total_likes = 0
            male_female_ratio = 0
            for gnl in gender_name_list:
                if gnl[0] in lnc[2]:
                    if gnl[1] == "Male":
                        male_count += 1
                    elif gnl[1] == "Female":
                        female_count += 1
                    total_likes += 1
                if gnl[0] == lnc[0]:
                    gender = gnl[1]

            if total_likes > 0:
                male_ratio = male_count / total_likes
            else:
                male_ratio = 0

            data_set.append((gender, lnc[1].encode("utf-8"), male_count, female_count, total_likes,
                             male_ratio))

        return data_set



m = MySQLDataExtractor()
print m.load_firstnames_from_sql_dump(m.load_firstnames_from_mysql())