
#__author__ = 'Henrik'

from __future__ import division
import mysql.connector
import csv
import os
import itertools
import data_helper
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter


def read_in_danish_names_from_files():

    drengenavne = []
    pigenavne = []
    unisexnavne = []
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    
    with open(os.path.join(data_dir, "drengenavne.csv"), "r") as in_file:
        for line in csv.reader(in_file):
            drengenavne.append(line[0].lower())

    with open(os.path.join(data_dir, "pigenavne.csv"), "r") as in_file:
        for line in csv.reader(in_file):
            pigenavne.append(line[0].lower())

    with open(os.path.join(data_dir, "unisexnavne.csv"), "r") as in_file:
        for line in csv.reader(in_file):
            unisexnavne.append(line[0].lower())

    return drengenavne, pigenavne, unisexnavne

def load_firstnames_from_sql_dump():

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

def find_gender_by_name(firstname, drengenavne, pigenavne, unisexnavne):

    gender_name_list = []

    ## Find the male names that are also unisex names
    #unisex_in_male = set(drengenavne).intersection(unisexnavne)

    # Find all male names that is used in the comments and labels it minus unisex names
    male_name_list = set(drengenavne).intersection(set(firstname)) - set(unisexnavne)

    for mnl in male_name_list:
        gender_name_list.append((mnl,'Male'))

    # Find all female names that is used in the comments and labels it minus unisex names
    female_name_list = set(pigenavne).intersection(set(firstname)) - set(unisexnavne)

    for fnl in female_name_list:
        gender_name_list.append((fnl,'Female'))

    return gender_name_list

def load_comments_from_database_and_combine_with_gender(gender_name_list):

    c_data_set = []
    gender_data_set = []
    name_list = []

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
                    name_list.append(name)
                    c_data_set.append(comment)
                    gender_data_set.append(gnl[1])

    cursor.close()
    cnx.close()

    return gender_data_set, c_data_set, name_list

def save_gender_with_comments_to_file(gender_data_set, c_data_set, filename):

    data_dir = os.path.join(os.path.dirname(__file__), '../data')

    training_set_file = filename+'.csv'

    if not os.path.isfile(training_set_file):
        with open(os.path.join(data_dir, training_set_file), "w") as out_file:
            out_file.write("")

    with open(os.path.join(data_dir, training_set_file), "wb") as f:
        writer = csv.writer(f)
        writer.writerows(itertools.izip(gender_data_set, c_data_set))

def load_gender_with_comments_from_file(filename):

    training_set_file = data_helper.get_data_file_path(filename+'.csv')

    trainingSet = []

    data_dir = os.path.join(os.path.dirname(__file__), '../data')

    with open(os.path.join(data_dir, training_set_file), 'r') as in_file:
        for line in csv.reader(in_file):
            trainingSet.append((line[1], line[0]))

    return trainingSet

drengenavne, pigenavne, unisexnavne = read_in_danish_names_from_files()
firstname = load_firstnames_from_sql_dump()
gender_name_list = find_gender_by_name(firstname, drengenavne, pigenavne, unisexnavne)
gender_data_set, c_data_set, name_list = load_comments_from_database_and_combine_with_gender(gender_name_list)
print len(name_list)
print name_list
print len(firstname)
save_gender_with_comments_to_file(gender_data_set,c_data_set,"testing")
trainingSet = load_gender_with_comments_from_file("testing")

cFemale = 0
cMale = 0
print trainingSet[0][1]
for t in trainingSet:
    if t[1] == "Female":
        cFemale += 1
    else:
        cMale += 1

total = cFemale + cMale
ratio_male_to_female = cMale/total
print ratio_male_to_female
print total
print cFemale, cMale

cnt = Counter(name_list).items()
sorted_cnt = sorted(cnt,key=itemgetter(1),reverse=True)

labels, values = zip(*sorted_cnt)

indexes = np.arange(len(labels))
width = 2

plt.bar(indexes, values, width)
#plt.xticks(indexes + width * 0.5, labels)
plt.show()

## likes of comments
# SELECT *, count(cf.f_uid) As favorites FROM Information.inf_dtu_user u, Information.inf_dtu_comment c,
#Information.inf_dtu_comment_flag cf
#where u.u_uid=c.c_uid and c.c_cid= cf.f_cid group by c.c_cid;

# json
# #with open(gender_and_comments_file, "w") as outfile:
# #    for re in zip(c_data_set,gender_data_set):
# #        json.dump({'gender':re[1], 'comment':re[0], }, outfile, indent=2)
#


