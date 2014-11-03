__author__ = 'Henrik'

import mysql.connector
import codecs
import csv

comment = []
firstName = []
gender = []
drengenavne = []

with open("/Users/Henrik/InfoMine/data/drengenavne.csv", "r") as in_file:
    for line in csv.reader(in_file):
         drengenavne.append(line[0].lower())


cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1',
                              database='Information')

cursor = cnx.cursor()
cursor1 = cnx.cursor()

query = ("SELECT u.u_name, c.c_body FROM Information.inf_dtu_user u, Information.inf_dtu_comment c where u.u_uid=c.c_uid")

cursor.execute(query)

for c in cursor:
    if c[0] != None:
        name = c[0]
        #firstName = name.split()[0].lower()
        firstName.append(name.split()[0].lower())
        #query2 = ("UPDATE Information.inf_dtu_user SET u_firstName=firstName")
        #cursor1.execute(query2)

        #for dn in drengenavne:
        #    if firstName == dn:

cursor.close()
cnx.close()

#with codecs.open("/Users/Henrik/InfoMine/data/drengenavne.csv", "r", "utf-8") as in_file:
#    for line in in_file.readlines():
#         drengenavne.append(line)

print drengenavne[1]
print len(drengenavne[1])
print firstName[1]
print len(firstName[1])
print type(drengenavne[1])
print type(firstName[1])

#for dn in drengenavne:
#    for fn in firstName:
#        if dn == fn:




#print gender[1]
