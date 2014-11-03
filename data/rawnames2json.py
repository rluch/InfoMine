#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import re

boy_names_dict = {}
boy_names_str = u""

girl_names_dict = {}
girl_names_str = u""

def parseNamesList(rawfilename):
    names_dict = {}
    names_str = u""
    with codecs.open(rawfilename, "r", "UTF-8") as names:
        for line in names:
            rawsplitup = line.split("\t")
            if len(rawsplitup) > 1:
                names_dict[rawsplitup[1]] = ""
    for name in names_dict:
        names_str += "%s, " % name

    return names_str

boys_csv = parseNamesList("boynames.raw")
girls_csv = parseNamesList("girlnames.raw")
print boys_csv
print girls_csv
