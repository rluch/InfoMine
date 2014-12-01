
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

    user_inf = []

    cnx = mysql.connector.connect(user='root', password='',
                          host='127.0.0.1',
                          database='Information')

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

def find_gender_by_name(user_inf, drengenavne, pigenavne, unisexnavne):

    gender_name_list = []
    firstname = []

    # Extract only names in to a list
    for ui in user_inf:
        firstname.append(ui[0])

    # Find all male names that is used in the comments and labels it minus unisex names
    male_name_list = set(drengenavne).intersection(set(firstname)) - set(unisexnavne)

    for mnl in male_name_list:
        gender_name_list.append((mnl,'Male'))

    female_name_list = set(pigenavne).intersection(set(firstname)) - set(unisexnavne)

    for fnl in female_name_list:
        gender_name_list.append((fnl,'Female'))

    return gender_name_list

def NEW_load(user_inf, gender_name_list):

    cnx = mysql.connector.connect(user='root', password='',
                          host='127.0.0.1',
                          database='Information')

    cursor = cnx.cursor()
    query = ("SELECT u.u_name, c.c_body, group_concat(cf.f_uid) As like_ids FROM Information.inf_dtu_user u, Information.inf_dtu_comment c,"
             "Information.inf_dtu_comment_flag cf where u.u_uid=c.c_uid and c.c_cid = f_cid group by c.c_cid;")
    cursor.execute(query)

    like_names_comments = []
    for c in cursor:
        if c[0] != None:
            name = c[0]
            firstName = name.split()[0].lower()
            comment = c[1]
            like_ids = c[2].split(",")

            #print like_ids
            #print type(like_ids[0])
            #for li in like_ids:
            #    li = int(li)
            #    if li in user_inf:
            #        print user_inf
            #print user_inf[1]
            like_names = []
            for ui in user_inf:
                if ui[1] in like_ids:
                    like_names.append(ui[0])

            like_names_comments.append([firstName, comment, like_names])

    #for gnl in gender_name_list:
    #    if gnl[0] == firstName:
    #        print ui[0]
    #

    for lnc in like_names_comments:
        male_count = 0
        female_count = 0
        total_likes = 0
        male_female_ratio = 0
        for gnl in gender_name_list:
            if gnl[0] in lnc[2]:
                #if gnl[0] == "Male":
                #    male_count += 1
                #elif gnl[0] == "Female":
                #    female_count += 1
                #total_likes += 1
                print gnl[1]
                #name_list.append(name)
                #c_data_set.append(comment)
                #gender_data_set.append(gnl[1])

    return like_names_comments


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

'''
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
ratio_male_to_female_1 = cFemale/total
print ratio_male_to_female_1
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
'''
user_inf = load_firstnames_from_sql_dump()
print user_inf[0]
drengenavne, pigenavne, unisexnavne = read_in_danish_names_from_files()
gender_name_list = find_gender_by_name(user_inf, drengenavne, pigenavne, unisexnavne)

like_names_comments = NEW_load(user_inf, gender_name_list)
print like_names_comments[1][2]

## likes of comments
# SELECT *, count(cf.f_uid) As favorites FROM Information.inf_dtu_user u, Information.inf_dtu_comment c,
#Information.inf_dtu_comment_flag cf
#where u.u_uid=c.c_uid and c.c_cid= cf.f_cid group by c.c_cid;

# json
# #with open(gender_and_comments_file, "w") as outfile:
# #    for re in zip(c_data_set,gender_data_set):
# #        json.dump({'gender':re[1], 'comment':re[0], }, outfile, indent=2)
#


