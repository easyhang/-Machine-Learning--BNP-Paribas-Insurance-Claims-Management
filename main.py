# This is the main module

import time
import classfication as c
import statistic as stat
from dataloader import loader
import Paint_statistic as paint


def classify():
    start = time.time()
    #c.Xgboost(dataset='normal')
    #c.Extremely_randomTree(missing=-999).doclassify()
    c.Knn(missing='mean')
    end = time.time()
    print end - start

def sta():
    dataset = loader('train.csv')
    stat.Categorical_stat().co_relation(dataset)
    #missing_stat = stat.Missing_stat().numofmissing_ofrow(dataset)
    #dp.List_dataProcess().fillMissingValueWithKnn(dataset, missing_stat, k=2)
    #stat.Categorical_stat().ca_label_proportion(dataset, type='times')

def plot():
    paint.painting()

if __name__ == '__main__':

    #sta()
    classify()
    #plot()






