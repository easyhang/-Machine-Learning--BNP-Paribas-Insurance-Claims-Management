'''
    This part is to analyze statistically the relevance of each feature and target. the distribution of target and each feature,
especially categorical feature. Based on this statistic output, we can have the weight of each value in each categorical
feature, the distance between each datapoint. Deeply, We can even guess some meaning of some features.
'''


import dataprocessing as dp
from numpy import *


class Categorical_stat():                           # analyze statistically based on categorical feature

    def count_stat(self, dataset, type='times', label=False):   # count the number of 0 and 1 in  each value of a categorical feature
        for (dataset_name, dataset_series) in dataset.iteritems():
            if not label:
                if dataset_series.dtype == 'O':
                    if type == 'times':
                        print dataset[dataset_name].value_counts()
                        print '\n'
                    if type == 'proportion':
                        print dataset[dataset_name].value_counts(1)
                        print '\n'
            if label:
                if dataset_series.dtype == 'O':
                    if type == 'times':
                        l = dataset[[dataset_name, 'target']]
                        print l.groupby('target')[dataset_name].value_counts()
                        print '\n'
                if dataset_series.dtype == 'O':
                    if type == 'proportion':
                        l = dataset[[dataset_name, 'target']]
                        lenth1 = 0
                        lenth2 = 50
                        while 1:
                            if lenth1 >= len(l.groupby('target')[dataset_name].value_counts(1)):
                                break
                            print l.groupby('target')[dataset_name].value_counts(1)[lenth1:lenth2]
                            lenth1 += 50
                            lenth2 += 50
                        print '\n'

    def ca_label_proportion(self, dataset, type='proportion'):          # count number of each value in categorical feature bease on label
        for (dataset_name, dataset_series) in dataset.iteritems():
                if dataset_series.dtype == 'O':
                    if type == 'proportion':
                        l = dataset[[dataset_name, 'target']]
                        print l.groupby(dataset_name)['target'].value_counts(1)
                        print '\n'
                    if type == 'times':
                        l = dataset[[dataset_name, 'target']]
                        print l.groupby(dataset_name)['target'].value_counts()
                        print '\n'

    def co_relation(self, dataset):                         # relation between each value of each feature
        ca = []
        for (dataset_name, dataset_series) in dataset.iteritems():
            if dataset_series.dtype == 'O':
                #dataset[dataset_name], perfeature_name = pd.factorize(dataset[dataset_name])
                ca.append(dataset_name)
        l = dataset[ca]
        for i in range(len(ca)):
            for j in range(i, len(ca)):
                print ca[j]
                print l.groupby(ca[i])[ca[j]].value_counts(1)
                print '\n'


class Missing_stat():                   # analyze statistically about missing value

    def numof_missing(self, dataset):       # to count number of missing value in each feature
        missnum = {}
        for (dataset_name, dataset_series) in dataset.iteritems():
         numbers = 0
         for num in dataset_series.isnull().iteritems():
             if num[1]:
                 numbers += 1
         missnum[dataset_name] = numbers
        print missnum

    def numofmissing_ofrow(self, dataset):      # to count number of missing value in each row
        filename = "missingResult.txt"
        dataset, target, id = dp.Pandas_dataProcess().label_extract(dataset, id=True)
        dataset = dp.Pandas_dataProcess().remove_categorical(dataset)
        dataset = dataset.values.tolist()
        missing_stat = {}
        with open(filename, "r+") as f:
            for (i, x) in enumerate(dataset):
                f.write(str(i))
                f.write(" : ")
                missing = 0
                for (j, num) in enumerate(x):
                    if not num < inf:
                        missing += 1
                        f.write(str(j)+", ")
                if missing not in missing_stat.keys():
                    missing_stat[missing] = [i]
                else:
                    missing_stat[missing].append(i)
                f.write("***|   |***")
                f.write(str(missing))
                f.write("\n")
        return missing_stat





