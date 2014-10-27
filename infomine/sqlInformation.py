# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 16:02:49 2014

@author: Henrik
"""

import sqlite3

con = sqlite3.connect('informationDB.db')
f = open('inf-dtu-demo.sql','r')
sql = f.read()
con.execute(sql)