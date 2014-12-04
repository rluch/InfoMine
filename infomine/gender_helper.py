# -*- coding: utf-8 -*-
"""
gender_helper.py
"""

from data_helper import get_data_file_path, load_and_return_lines_from_csv_file


class GenderHelper(object):
    """
    GenderHelper contains methods for quickly identifying the gender
    of a given name, based upon lists of male, female and unisex names
    """
    def __init__(self):
        self.malenames = []
        self.femalenames = []
        self.unisexnames = []
        self.loaded_names_count = {'male': 0, 'female': 0, 'unisex': 0}
        self.preload_names()
        self.filter_out_unisexnames()

    def lower_names_list(self, list_of_names):
        return [fname[0].lower() for fname in list_of_names]

    def load_malenames_from_file(self):
        names = load_and_return_lines_from_csv_file(
            get_data_file_path("drengenavne.csv"))
        self.malenames = self.lower_names_list(names)

    def load_femalenames_from_file(self):
        names = load_and_return_lines_from_csv_file(
            get_data_file_path("pigenavne.csv"))
        self.femalenames = self.lower_names_list(names)

    def load_unisexnames_from_file(self):
        names = load_and_return_lines_from_csv_file(
            get_data_file_path("unisexnavne.csv"))
        self.unisexnames = self.lower_names_list(names)

    def preload_names(self):
        self.load_malenames_from_file()
        self.load_femalenames_from_file()
        self.load_unisexnames_from_file()

    def filter_out_unisexnames(self):
        self.loaded_names_count['male'] = len(self.malenames)
        self.loaded_names_count['female'] = len(self.femalenames)
        self.malenames = set(self.malenames).difference(set(self.unisexnames))
        self.femalenames = set(
            self.femalenames).difference(set(self.unisexnames))

    def get_gender_by_name(self, fname):
        if fname in self.malenames:
            return 'male'
        elif fname in self.femalenames:
            return 'female'
        else:
            return 'unknown'

    def print_stats(self):
        print("Loaded male names:\t%i" % self.loaded_names_count['male'])
        print("Loaded female names:\t%i" % self.loaded_names_count['female'])
        print("Loaded unisex names:\t%i" % len(self.unisexnames))
        print("Filtered male names:\t%i" % len(self.malenames))
        print("Filtered female names:\t%i" % len(self.femalenames))
