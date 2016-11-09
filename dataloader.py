'''Iuput .csv data from current file fold
   Input data files are not available in other directory.
'''

from pandas import read_csv


def loader(filepath):
    dataframe = read_csv(filepath)    #loaddata from .csv fild
    return dataframe



