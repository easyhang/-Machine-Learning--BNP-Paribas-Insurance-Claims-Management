'''
 This file would do process the data from BNP scoure
 The data-preprocessing method include that deal with missing value, factorize categorical value, convert type of data,
shuffle and sort the data, as well as increasing and reducing the dimension of the dataset
'''


import pandas as pd                                     #data framework
from numpy import *                                     #linear algebra
from sklearn.neighbors import NearestNeighbors          #Knn method process missing value
import writetxt as writetxt
import collections                                      # sort function
import statistic as stat                                # statistic data
from sklearn import metrics, linear_model               # linear regression
import random                                           # random generator


class List_dataProcess():                              # process list type data including split dataset to missing part and normal part

    def missvalue_arrange(self, dataset):
        for i in range(len(dataset)):
            for j in range(len(dataset[i])):
                if dataset[i][j] < inf:                 # judge whether it is missing value in list
                    pass
                else:
                    dataset[i][j] = None
        return dataset

    def missingset_split(self, dataset):                #This function is to split the dataset into two parts which one has lots of missing value
        miss = []
        notmiss = []
        for row in dataset:
            missnum = 0
            for col in row:
                if not col < inf:
                    missnum += 1
            if missnum > 50:                            # if number of missing value in this datapoint bigger than 50, this point will be seen as a missing point
                miss.append(row)
            else:
                notmiss.append(row)
        return miss, notmiss

    def label_extract(self, dataset):                   # Extract the label from the dataset
        data = []
        label = []
        for i in dataset:
            label.append(i[0])
            data.append(i[1:])
        return data, label

    def check_missing(self, data):                      # check whether this data point is a missing part or not
        misnum = 0
        for i in data:
            if not i < inf:
                misnum += 1
        if misnum > 50:
            return True
        return False

    def fillMissingValueWithKnn(self, dataset, missing_stat, k=10):                     # fill the list type data by using knn method
        dataset, target, id = Pandas_dataProcess().label_extract(dataset, id=True)
        dataset = Pandas_dataProcess().remove_categorical(dataset).values.tolist()
        fulldata = []
        for x in missing_stat[0]:
            fulldata.append(dataset[x])
            # writetxt.writer("nonMissingValue.txt", dataset[x])                        # the k nearest neigbhor index of dataport will save in a txt
        fulldata = array(fulldata)
        for index in missing_stat[1]:
            test = array(dataset[index])
            result = []
            missing_index = []
            for x in fulldata:
                dis = x - test
                # print max(dis)
                distance = 0
                for i, n in enumerate(dis):
                    if n*n < inf:
                        distance += n*n
                    else:
                        missing_index.append(i)
                result.append(distance)
            fulldata[missing_stat[1][index],[missing_index]] = fulldata[argsort(result)[0]][missing_index]
            writetxt.writer("nonMissingValue.txt", missing_stat[index][index])


class Pandas_dataProcess():                                 # process Pandas framework type data including split dataset to missing part and normal part

    def __init__(self):                                     # name of categorical features
        self.categorical_set = ['v3', 'v22', 'v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110', 'v112', 'v113', 'v125']
        #self.categorical_set = ['v3', 'v22', 'v30', 'v31', 'v47', 'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v107', 'v110', 'v112', 'v113', 'v125']

    def label_extract(self, dataset, id=False, istest=False, real_extract=True):            # Extract label or id from dataframework
        if id:
            id = dataset['ID']
            if istest:
                dataset = dataset.drop('ID', axis=1)
                return dataset, id
            else:
                labels = dataset['target']
                dataset = dataset.drop(['ID', 'target'], axis=1)
            return dataset, labels, id
        else:
            labels = dataset['target']
            if real_extract:
                dataset = dataset.drop(['ID', 'target'], axis=1)
            return dataset, labels

    def id_droper(self, dataset):                           # Remove ID#
        dataset = dataset.drop('ID', axis=1)
        return dataset

    def remove_categorical(self, dataset):                  # Remove categorical feature from data framework
        for ca in self.categorical_set:
            dataset = dataset.drop(ca, axis=1)
        return dataset

    def fill_missvalue(self, dataset, value='mean', consider_cat=True, label_based=False):  # methods of dealing with missing value
        if value == 'linear_knn':                               # select k-nearest neigbors and using them to do linear_regression to fill the missing
            catagorical_set = [3, 22, 24, 30, 31, 47, 52, 56, 66, 71, 74, 75, 79, 91, 107, 110, 112, 113, 125]
            for i in range(len(catagorical_set)):
                catagorical_set[i] -= 1

            missing_stat = stat.Missing_stat().numofmissing_ofrow(dataset)
            nonMissingValSet = []
            missingIndex = missing_stat[0]
            dataset, target, id = Pandas_dataProcess().label_extract(dataset, id=True)
            dataset = dataset.values.tolist()
            for i in missingIndex:
                nonMissingValSet.append(dataset[i])
            print '1'

            ########################################## KNN ##########################
            K = 10
            filledSet = []
            for record in dataset:
                nonMissTmp = copy(nonMissingValSet)
                missedIndex = []
                fullSetDic = {}
                for index in range(len(record)):
                    if record[index] == '':
                        missedIndex.append(index)
                for index in range(len(nonMissTmp)):
                    distance = 0.0
                    line = nonMissTmp[index]
                    for i in range(len(line)):
                        if i in missedIndex or i in catagorical_set:
                            continue
                        else:
                            distance += pow((float(record[i]) - float(line[i])), 2)
                    distance = sqrt(distance)
                    fullSetDic[distance] = []
                    fullSetDic[distance].extend(nonMissTmp[index])
                od = collections.OrderedDict(sorted(fullSetDic.items()))
                topK = []
                for i in range(K):
                    key, value = fullSetDic.popitem()
                    topK.append(value)
                print '2'

                ########################################## Linear regression ########################
                fillIn = []
                train_x = []
                for row in range(len(topK)):
                    train_x.append([])
                    for col in range(len(topK[0])):
                        if col in missedIndex:
                            continue
                        else:
                            train_x[-1].append(topK[row][col])
                test_x = []
                for i in range(len(record)):
                    if record[i] == '':
                        continue
                    else:
                        test_x.append(record[i])
                for i in range(len(record)):
                    if record[i] == '':
                        train_y = []
                        for row in range(len(topK)):
                            train_y.append(topK[row][i])
                        regr = linear_model.LinearRegression()
                        regr.fit(train_x, train_y)
                        predict_label = regr.predict(test_x)
                        fillIn.append(predict_label)
                    else:
                        fillIn.append(record[i])
                filledSet.append(fillIn)
                print '3'
            return filledSet
        if value == 'nn':                       # choose the nearest neighbor value to fill missing value
            copydataset = dataset
            copydataset = self.remove_categorical(copydataset)
            copydataset = self.fill_missvalue(copydataset, consider_cat=False)
            listset = array(copydataset.values.tolist())
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(listset)
            distances, indices = nbrs.kneighbors(listset)
            print indices
            a = 0
            for (dataset_name, dataset_series) in dataset.iteritems():
                print a
                a += 1
                nan_num = len(dataset[dataset_series.isnull()])
                if nan_num > 0:
                    for index, values in dataset_series.iteritems():
                        if not dataset.ix[index, dataset_name] < inf:
                            dataset.ix[index, dataset_name] = dataset.ix[indices[index][1], dataset_name]
            print dataset
            return dataset
        t = 1
        if label_based:
            dataset = dataset.sort(['target'], ascending=False)
        for (dataset_name, dataset_series) in dataset.iteritems():
            print t
            t += 1
            if value == 'mean':                                # use mean value to fill the missing value
                nan_num = len(dataset[dataset_series.isnull()])
                if nan_num > 0:
                    if label_based:
                        if consider_cat:
                            if dataset_name in self.categorical_set:
                                for (values, times) in dataset[dataset_name].value_counts().iteritems():
                                    mode = values
                                    break
                                dataset.loc[dataset_series.isnull(), dataset_name] = mode
                            else:
                                    labelbased_mean = []
                                    for (dataset_name2, dataset_series2) in dataset.groupby('target'):
                                        labelbased_mean.append(dataset_series2[dataset_name].mean())
                                    dataset[:87022].loc[dataset_series.isnull(), dataset_name] = labelbased_mean[1]
                                    dataset[87022:].loc[dataset_series.isnull(), dataset_name] = labelbased_mean[0]
                        else:
                            dataset.loc[dataset_series.isnull(), dataset_name] = dataset_series.mean()
                    else:
                        if consider_cat:
                            if dataset_name in self.categorical_set:
                                for (values, times) in dataset[dataset_name].value_counts().iteritems():
                                    mode = values
                                    break
                                dataset.loc[dataset_series.isnull(), dataset_name] = mode
                            else:
                                dataset.loc[dataset_series.isnull(), dataset_name] = dataset_series.mean()
                        else:
                            dataset.loc[dataset_series.isnull(), dataset_name] = dataset_series.mean()
            elif value == 'std':                    # use standard deviation to fill the missing value
                nan_num = len(dataset[dataset_series.isnull()])
                if nan_num > 0:
                    if consider_cat:
                        if dataset_name in self.categorical_set:
                            for (values, times) in dataset[dataset_name].value_counts().iteritems():
                                mode = values
                                break
                            dataset.loc[dataset_series.isnull(), dataset_name] = mode
                        else:
                            dataset.loc[dataset_series.isnull(), dataset_name] = dataset_series.std()
                    else:
                        dataset.loc[dataset_series.isnull(), dataset_name] = dataset_series.std()
            elif value == 'mode':                   # use the mode to fill the missing value
                nan_num = len(dataset[dataset_series.isnull()])
                if nan_num > 0:
                    for (values, times) in dataset[dataset_name].value_counts().iteritems():
                        mode = values
                        break
                    dataset.loc[dataset_series.isnull(), dataset_name] = mode
            else:
                nan_num = len(dataset[dataset_series.isnull()])
                if nan_num > 0:
                    dataset.loc[dataset_series.isnull(), dataset_name] = value
        return dataset

    def trans_categorical(self, dataset, istest=False, feature_name=None, normalized=False):   # factorize categorical values
        if istest:
            i = 0
            for (dataset_name, dataset_series) in dataset.iteritems():
                if dataset_series.dtype == 'O':
                    dataset[dataset_name] = feature_name[i].get_indexer(dataset[dataset_name])
                    i += 1
            return dataset
        else:
            feature_name = []
            for (dataset_name, dataset_series) in dataset.iteritems():
                if dataset_series.dtype == 'O':
                    dataset[dataset_name], perfeature_name = pd.factorize(dataset[dataset_name])
                    feature_name.append(perfeature_name)
        return dataset, feature_name

    def trans_topanda(self, data):                              # convert list data to Pandas data framework
        featurename = ['v' + str(i+1) for i in range(130)]
        featurename.insert(0, 'target')
        return pd.DataFrame(data, columns=featurename)


class AddNearestNeighbourLinearFeatures:                # This class is to increase the dimission of dataset based on the features which are linear based

    def __init__(self, n_neighbours=1, max_elts=3, verbose=True, random_state=12):
        self.rnd = random_state
        self.n =n_neighbours
        self.max_elts=max_elts
        self.verbose=verbose
        self.neighbours=[]
        self.clfs=[]

    def fit(self, train,y):                             # fit the label
        if self.rnd != None:
            random.seed(self.rnd)
        if self.max_elts == None:
            self.max_elts = len(train.columns)
        list_vars=list(train.columns)
        random.shuffle(list_vars)

        lastscores = zeros(self.n)+1e15

        for elt in list_vars[:self.n]:
            self.neighbours.append([elt])
        list_vars = list_vars[self.n:]

        for elt in list_vars:
            indice=0
            scores=[]
            for elt2 in self.neighbours:
                if len(elt2)<self.max_elts:
                    clf = linear_model.LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1)
                    clf.fit(train[elt2+[elt]], y)
                    scores.append(metrics.log_loss(y, clf.predict(train[elt2 + [elt]])))
                    indice = indice+1
                else:
                    scores.append(lastscores[indice])
                    indice = indice+1
            gains = lastscores - scores
            if gains.max() > 0:
                temp=gains.argmax()
                lastscores[temp] = scores[temp]
                self.neighbours[temp].append(elt)

        indice=0
        for elt in self.neighbours:
            clf=linear_model.LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1)
            clf.fit(train[elt], y)
            self.clfs.append(clf)
            if self.verbose:
                print(indice, lastscores[indice], elt)
            indice=indice+1

    def transform(self, train):
        indice=0
        for elt in self.neighbours:
            train['_'.join(pd.Series(elt).sort_values().values)]=self.clfs[indice].predict(train[elt])
            indice=indice+1
        return train

    def fit_transform(self, train, y):
        self.fit(train, y)
        return self.transform(train)



















