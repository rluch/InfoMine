
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
            like_names = []
            for ui in user_inf:
                if ui[1] in like_ids:
                    like_names.append(ui[0])

            like_names_comments.append([firstName, comment, like_names])

    data_set = []
    data_set.append(("Gender", "Comment", "Number_of_male_likes", "Number_of_female_likes", "Total_likes",
                     "Male_likes_compared_to_Female"))

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


def save_gender_with_comments_to_file(data_set, filename):

    data_dir = os.path.join(os.path.dirname(__file__), '../data')

    data_set_file = filename+'.csv'

    if not os.path.isfile(data_set_file):
        with open(os.path.join(data_dir, data_set_file), "w") as out_file:
            out_file.write("")

    with open(os.path.join(data_dir, data_set_file), "wb") as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(data_set)
        #writer = csv.writer(f)
        #writer.writerows(itertools.izip(gender_data_set, c_data_set))

def load_gender_with_comments_from_file(filename):

    data_set_file = data_helper.get_data_file_path(filename+'.csv')

    data_set = []

    data_dir = os.path.join(os.path.dirname(__file__), '../data')

    with open(os.path.join(data_dir, data_set_file), 'r') as in_file:
        for line in csv.reader(in_file):
            data_set.append((line[0].decode("utf-8"), line[1].decode("utf-8"), line[2], line[3], line[4], line[5]))

    return data_set


user_inf = load_firstnames_from_sql_dump()
print user_inf[0]
drengenavne, pigenavne, unisexnavne = read_in_danish_names_from_files()
gender_name_list = find_gender_by_name(user_inf, drengenavne, pigenavne, unisexnavne)

data_set = NEW_load(user_inf, gender_name_list)

print type(data_set[0][0])
save_gender_with_comments_to_file(data_set,"testingNew")
data_set_loaded = load_gender_with_comments_from_file("testingNew")
print data_set_loaded[0:2]
print type(data_set[0][0])

cFemale = 0
cMale = 0
print data_set[0][1]
for t in data_set:
    if t[0] == "Female":
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

'''
cnt = Counter(name_list).items()
sorted_cnt = sorted(cnt,key=itemgetter(1),reverse=True)

labels, values = zip(*sorted_cnt)

indexes = np.arange(len(labels))
width = 2

plt.bar(indexes, values, width)
#plt.xticks(indexes + width * 0.5, labels)
plt.show()
'''

# json
# #with open(gender_and_comments_file, "w") as outfile:
# #    for re in zip(c_data_set,gender_data_set):
# #        json.dump({'gender':re[1], 'comment':re[0], }, outfile, indent=2)
#


