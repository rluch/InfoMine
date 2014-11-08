__author__ = 'Henrik'

import mysql.connector
import csv
import json
import os
import pprint

firstName = []
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


cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1',
                              database='Information')

cursor = cnx.cursor()
query = ("SELECT u_name FROM Information.inf_dtu_user")
cursor.execute(query)

for c in cursor:
    if c[0] != None:
        name = c[0]
        firstName.append(name.split()[0].lower())

gender_name_list = []
firstName = set(firstName)
for fn in firstName:
    for dn in drengenavne:
        if dn == fn:
            gender_name_list.append((fn,'Male'))
    for pn in pigenavne:
        if pn == fn:
            gender_name_list.append((fn,'Female'))
    #for un in unisexnavne:
    #    if un == fn:
    #        gender_name_list.append((fn,'Unisex'))

c_data_set = []
gender_data_set = []

query = ("SELECT u.u_name, c.c_body FROM Information.inf_dtu_user u, Information.inf_dtu_comment c where u.u_uid=c.c_uid")
cursor.execute(query)

for c in cursor:
    if c[0] != None:
        name = c[0]
        comment = c[1]
        firstName = name.split()[0].lower()
        for gnl in gender_name_list:
            if gnl[0] == firstName:
                c_data_set.append(comment)
                gender_data_set.append(gnl[1])


cursor.close()
cnx.close()

gender_and_comments_file = "gender_and_comments.json"

if not os.path.isfile(gender_and_comments_file):
        with open(gender_and_comments_file, "w") as out_file:
            out_file.write("")

with open(gender_and_comments_file, "w") as outfile:
    json.dump({'gender':gender_data_set, 'comment':c_data_set, }, outfile, indent=2)

with open(gender_and_comments_file) as data_file:
    data = json.load(data_file)




