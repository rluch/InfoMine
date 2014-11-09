__author__ = 'Henrik'

import mysql.connector
import csv
import os
import itertools

class DataCollection():

    def read_in_danish_names_from_files(self):

        drengenavne = []
        pigenavne = []
        unisexnavne = []

        with open("/Users/Henrik/InfoMine/data/drengenavne.csv", "r") as in_file:
            for line in csv.reader(in_file):
                drengenavne.append(line[0].lower())

        with open("/Users/Henrik/InfoMine/data/pigenavne.csv", "r") as in_file:
            for line in csv.reader(in_file):
                pigenavne.append(line[0].lower())

        #with open("/Users/Henrik/InfoMine/data/unisexnavne.csv", "r") as in_file:
        #    for line in csv.reader(in_file):
        #        unisexnavne.append(line[0].lower())

        return drengenavne, pigenavne

    def load_names_from_sql_dump(self):

        firstname = []

        cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1',
                              database='Information')

        cursor = cnx.cursor()
        query = ("SELECT u_name FROM Information.inf_dtu_user")
        cursor.execute(query)

        for c in cursor:
            if c[0] != None:
                name = c[0]
                firstname.append(name.split()[0].lower())

        cursor.close()
        cnx.close()

        return firstname

    def find_gender_by_name(self, firstname, drengenavne, pigenavne):

        gender_name_list = []
        firstname = set(firstname)
        for fn in firstname:
            for dn in drengenavne:
                if dn == fn:
                    gender_name_list.append((fn,'Male'))
            for pn in pigenavne:
                if pn == fn:
                    gender_name_list.append((fn,'Female'))
            #for un in unisexnavne:
                #if un == fn:
#                   #gender_name_list.append((fn,'Unisex'))

        return gender_name_list

    def load_comments_from_database_and_combine_with_gender(self, gender_name_list):

        c_data_set = []
        gender_data_set = []

        cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1',
                              database='Information')

        cursor = cnx.cursor()
        query = ("SELECT u.u_name, c.c_body FROM Information.inf_dtu_user u, Information.inf_dtu_comment c where u.u_uid=c.c_uid")
        cursor.execute(query)

        for c in cursor:
            if c[0] != None:
                name = c[0]
                comment = c[1].encode("utf-8")
                firstName = name.split()[0].lower()
                for gnl in gender_name_list:
                    if gnl[0] == firstName:
                        c_data_set.append(comment)
                        gender_data_set.append(gnl[1])

        cursor.close()
        cnx.close()

        return gender_data_set, c_data_set

    def save_gender_with_comments_to_file(self, gender_data_set, c_data_set):

        gender_and_comments_file = "gender_and_comments.csv"

        if not os.path.isfile(gender_and_comments_file):
                 with open(gender_and_comments_file, "w") as out_file:
                     out_file.write("")

        with open(gender_and_comments_file, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(itertools.izip(gender_data_set, c_data_set))

    def load_gender_with_comments_from_file(self):

        gender_and_comments_file = "gender_and_comments.csv"

        trainingSet = []

        with open(gender_and_comments_file, "r") as in_file:
            for line in csv.reader(in_file):
                trainingSet.append((line[0],line[1]))

        return trainingSet

    def __init__(self):
        firstname = self.load_names_from_sql_dump()
        drengenavne, pigenavne = self.read_in_danish_names_from_files()

dc = DataCollection()
drengenavne, pigenavne = dc.read_in_danish_names_from_files()
firstname = dc.load_names_from_sql_dump()
gender_name_list = dc.find_gender_by_name(firstname, drengenavne, pigenavne)
gender_data_set, c_data_set = dc.load_comments_from_database_and_combine_with_gender(gender_name_list)
dc.save_gender_with_comments_to_file(gender_data_set,c_data_set)
trainingSet = dc.load_gender_with_comments_from_file()

print trainingSet[0]

# json
# #with open(gender_and_comments_file, "w") as outfile:
# #    for re in zip(c_data_set,gender_data_set):
# #        json.dump({'gender':re[1], 'comment':re[0], }, outfile, indent=2)
#


