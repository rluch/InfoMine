__author__ = 'Henrik'

import mysql.connector
import csv
import os
import itertools
import codecs


class DataCollection:

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

        with open("/Users/Henrik/InfoMine/data/unisexnavne.csv", "r") as in_file:
            for line in csv.reader(in_file):
                unisexnavne.append(line[0].lower())

        return drengenavne, pigenavne, unisexnavne

    def load_firstnames_from_sql_dump(self):

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

    def find_gender_by_name(self, firstname, drengenavne, pigenavne, unisexnavne):

        gender_name_list = []

        # Find the male names that are also unisex names
        unisex_in_male = set(drengenavne).intersection(unisexnavne)

        # Find all male names that is used in the comments and labels it minus unisex names
        male_name_list = set(drengenavne).intersection(set(firstname)) - unisex_in_male

        for mnl in male_name_list:
            gender_name_list.append((mnl,'Male'))

        # Find the female names that are also unisex names
        unisex_in_female = set(pigenavne).intersection(unisexnavne)

        # Find all female names that is used in the comments and labels it minus unisex names
        female_name_list = set(pigenavne).intersection(set(firstname)) - unisex_in_female

        for fnl in female_name_list:
            gender_name_list.append((fnl,'Female'))

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

    def save_gender_with_comments_to_file(self, gender_data_set, c_data_set, filename):

        training_set_file = filename+'.csv'

        if not os.path.isfile(training_set_file):
                 with open(training_set_file, "w") as out_file:
                     out_file.write("")

        with open(training_set_file, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(itertools.izip(gender_data_set, c_data_set))

    def load_gender_with_comments_from_file(self, filename):

        training_set_file = filename+'.csv'

        trainingSet = []

        with open(training_set_file, "r") as in_file:
            for line in csv.reader(in_file):
                trainingSet.append((line[1], line[0]))

        return trainingSet

    #def __init__(self, training_set):
    #    trainingSet = self.load_gender_with_comments_from_file("testing")
    #    self.training_set = trainingSet

dc = DataCollection()
drengenavne, pigenavne, unisexnavne = dc.read_in_danish_names_from_files()
firstname = dc.load_firstnames_from_sql_dump()
gender_name_list = dc.find_gender_by_name(firstname, drengenavne, pigenavne, unisexnavne)
gender_data_set, c_data_set = dc.load_comments_from_database_and_combine_with_gender(gender_name_list)
dc.save_gender_with_comments_to_file(gender_data_set,c_data_set,"testing")
trainingSet = dc.load_gender_with_comments_from_file("testing")

#print trainingSet[0]

#dc1 = DataCollection().load_gender_with_comments_from_file("testing")
#print dc1[0]

# json
# #with open(gender_and_comments_file, "w") as outfile:
# #    for re in zip(c_data_set,gender_data_set):
# #        json.dump({'gender':re[1], 'comment':re[0], }, outfile, indent=2)
#


