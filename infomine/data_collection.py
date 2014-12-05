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
import random


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
    """ Moved to data_helper"""
    data_set_file = data_helper.get_data_file_path(filename+'.csv')

    data_set = []

    data_dir = os.path.join(os.path.dirname(__file__), '../data')

    with open(os.path.join(data_dir, data_set_file), 'r') as in_file:
        for line in csv.reader(in_file):
            data_set.append((line[0].decode("utf-8"), line[1].decode("utf-8"), line[2], line[3], line[4], line[5]))

    return data_set


def remove_male_samples_from_data_set(data_set):

    # Remove column names and sorts the list
    data_set_sorted = sorted(data_set[1:])

    # Extract all female and male to a seperat list
    data_set_female = data_set_sorted[:1130]
    data_set_male = data_set_sorted[1130:]

    # Random remove some of the male samples
    random.shuffle(data_set_male)
    data_set_male = data_set_male[:1370]

    # Merge the new male list and female list and shuffle
    new_data_set = data_set_female + data_set_male
    random.shuffle(new_data_set)

    return new_data_set

def count_number_of_male_and_female_comments(data_set):

    cFemale = 0
    cMale = 0
    sum_male_likes = 0
    sum_female_likes = 0
    sum_total_likes = 0

    print data_set[0]
    for t in data_set:
        if t[0] == "Female":
            cFemale += 1
            sum_female_likes += int(t[3])
        else:
            cMale += 1
            sum_male_likes += int(t[2])

        sum_total_likes += int(t[4])

    total = cFemale + cMale
    procent_male = cMale/total
    procent_female = cFemale/total
    print "Procent male: %s" %procent_male
    print "Procent female: %s" %procent_female
    print "Total comments: %s" %total
    print "Male comments: %s" %cFemale
    print "Female comments: %s" %cMale

    avg_male_likes = sum_male_likes/cMale
    avg_female_likes = sum_female_likes/cFemale
    avg_total_likes = sum_total_likes/total
    print "average male likes pr comment: %s" %avg_male_likes
    print "average female likes pr comment: %s" %avg_female_likes
    print "average total likes pr comment: %s" %avg_total_likes

    return None


user_inf = load_firstnames_from_sql_dump()

drengenavne, pigenavne, unisexnavne = read_in_danish_names_from_files()
gender_name_list = find_gender_by_name(user_inf, drengenavne, pigenavne, unisexnavne)

data_set = load_comments_and_gender_and_comment_likes_from_sql_dump(user_inf, gender_name_list)

data_set = remove_male_samples_from_data_set(data_set)

print data_set[1]
save_gender_with_comments_to_file(data_set,"testingNew")
data_set_loaded = load_gender_with_comments_from_file("testingNew")

count_number_of_male_and_female_comments(data_set)

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



