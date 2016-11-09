'''
    In this part, We have 6 classification methods to train the model including Xgboost, Extremely random tree, SVM,
Knn, naive bayes and logistic regression. we will have the logloss or accuracy rate as to evaluate the how good about the model is.
Also, We plot the ROC and AUC in each method in this module
'''


from dataloader import loader
import dataprocessing as dp
import xgboost as xgb
from datawriter import writer
from numpy import *
from sklearn.naive_bayes import BernoulliNB
from sklearn import ensemble, svm
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


class Xgboost:                                         # Xgboost

    def __init__(self, dataset='normal', missing='mean'):
        self.missing = missing
        if dataset == 'split':
            self.split()
            self.doclassify(type=dataset)
        if dataset == 'normal':
            self.normal()
            self.doclassify()

    def split(self):                                    # process if the dataset split into two parts, which contains lots of missing value
        self.train = loader('train.csv')
        self.test = loader('test.csv')
        self.test, self.id = dp.Pandas_dataProcess().label_extract(self.test, id=True, istest=True)
        self.train, feature_name = dp.Pandas_dataProcess().trans_categorical(self.train)
        self.test = dp.Pandas_dataProcess().trans_categorical(self.test, feature_name=feature_name, istest=True)
        self.train = dp.Pandas_dataProcess().id_droper(self.train)
        self.train = self.train.values.tolist()
        self.test = self.test.values.tolist()
        self.misstrain, self.normaltrain = dp.List_dataProcess().missingset_split(self.train)
        self.misstrain, self.misstrain_y = dp.List_dataProcess().label_extract(self.misstrain)
        self.normaltrain, self.normaltrain_y = dp.List_dataProcess().label_extract(self.normaltrain)
        self.misstest, self.normaltest = dp.List_dataProcess().missingset_split(self.test)
        self.normaltrain = dp.Pandas_dataProcess().trans_topanda(self.normaltrain)
        self.normaltrain = dp.Pandas_dataProcess().fill_missvalue(self.normaltrain, value=self.missing)
        self.normaltest = dp.Pandas_dataProcess().trans_topanda(self.normaltest)
        self.normaltest = dp.Pandas_dataProcess().fill_missvalue(self.normaltest, value=self.missing)

    def normal(self):                                   # process if the dataset not split into two parts
        self.train = loader('train.csv')
        self.train_x, self.train_y = dp.Pandas_dataProcess().label_extract(self.train, real_extract=False)
        self.test = loader('test.csv')
        self.test, self.id = dp.Pandas_dataProcess().label_extract(self.test, id=True, istest=True)
        self.train_x, feature_name = dp.Pandas_dataProcess().trans_categorical(self.train_x)
        self.test = dp.Pandas_dataProcess().trans_categorical(self.test, feature_name=feature_name, istest=True)
        self.train_x = dp.Pandas_dataProcess().fill_missvalue(self.train_x, value=0)
        self.test = dp.Pandas_dataProcess().fill_missvalue(self.test, value=0)
        self.train_x = self.train_x.iloc[random.permutation(len(self.train_x))]
        self.train_x, self.train_y = dp.Pandas_dataProcess().label_extract(self.train_x)

    def doclassify(self, type='normal'):                # Boosting
        if type == 'split':
            dtrainmis = xgb.DMatrix(array(self.misstrain), array(self.misstrain_y), missing=NAN)
            dtest = xgb.DMatrix(array(self.normaltest), missing=NAN)
            dtestmis = xgb.DMatrix(array(self.misstest), missing=NAN)
            param = {'bst:max_depth':10, 'bst:eta':0.02, 'silent':1, 'objective':'binary:logistic', 'subsample':0.8,"colsample_bytree": 0.68,"booster": "gbtree"}
            param['nthread'] = 4
            param['eval_metric'] = 'logloss'
            evallist  = [(dtrainmis, 'train')]
            num_round = 320
            bstmis = xgb.train(param, dtrainmis, num_round, evallist,)
            dtrain = xgb.DMatrix(array(self.normaltrain), array(self.normaltrain_y))
            num_round = 366
            evallist  = [(dtrain, 'train')]
            bst = xgb.train(param, dtrain, num_round, evallist,)
            ypredmis = bstmis.predict(dtestmis)
            ypred = bst.predict(dtest)
            result = []
            output1 = list(ypredmis)
            output2 = list(ypred)
            for i in self.test:
                if dp.List_dataProcess().check_missing(i):
                    result.append(output1.pop(0))
                else:
                    result.append(output2.pop(0))
            print len(output1)
            print len(output2)
            writer(self.id, result)

        if type == 'normal':
            dtrain = xgb.DMatrix(array(self.train_x), array(self.train_y))
            dtest = xgb.DMatrix(array(self.test))
            param = {'bst:max_depth':10, 'bst:eta':0.02, 'silent':1, 'objective':'binary:logistic', 'subsample':0.9,"colsample_bytree": 0.68,"booster": "gbtree"}
            param['nthread'] = 4
            param['eval_metric'] = 'logloss'
            evallist  = [(dtrain, 'train')]
            num_round = 300
            bst = xgb.train(param, dtrain, num_round, evallist,)
            ypred = bst.predict(dtest)
            writer(self.id, ypred)

            acc = 0.0
            for i in range(10000):
                if array(self.train_y)[len(self.train_y)-10000+i] == 1 and ypred[i] > 0.35:
                    acc += 1
                if array(self.train_y)[len(self.train_y)-10000+i] == 0 and ypred[i] <= 0.35:
                    acc += 1
            print "Accuracy : ", acc/10000
            fpr, tpr, thresholds = metrics.roc_curve(self.train_y[-10000:], ypred, pos_label=1)
            for i in range(len(fpr)):
                plt.plot(fpr[i], tpr[i], "b*")
                plt.plot(fpr, tpr)
            plt.title(val)
            plt.show()
            print "AUC : ", metrics.auc(fpr, tpr)
            print thresholds


class Naive(Xgboost):                           # Naive Bayes classification

    def doclassify(self, type='normal'):
        if type == 'normal':
            clf = BernoulliNB()
            clf.fit(self.train_x, self.train_y)
            BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
            score = clf.score(self.train_x, self.train_y)
            print 'score = ', score


class Svm(Xgboost):                             # Support vector machine classification

    def doclassify(self, type='normal'):
        self.train_x = self.train_x.values.tolist()
        self.train_y = self.train_y.values.tolist()
        clf = svm.SVR(C=0.2)
        clf.fit(array(self.train_x[:-10000]), array(self.train_y[:-10000]))
        print clf.score(self.train_x[-10000:], self.train_y[-10000:])


class Knn(Xgboost):                             # K-nearest neighbor classification

    def doclassify(self, type='normal'):
        self.train_x = self.train_x.values.tolist()
        self.train_y = self.train_y.values.tolist()
        self.test = self.test.values.tolist()
        neigh = KNeighborsClassifier(n_neighbors=15)
        neigh.fit(self.train_x, self.train_y)
        t = self.train_y[-10000:]
        o = neigh.predict_proba(self.test)
        a = []
        for ele in o:
            a.append(ele[1])
        writer(self.id, a)


class Logistic(Xgboost):                        # Logistic Regression

    def doclassify(self, type='normal'):
        self.train_x = self.train_x.values.tolist()
        self.train_y = self.train_y.values.tolist()
        clf = LogisticRegression(C=0.2)
        clf.fit(array(self.train_x[:-10000]), array(self.train_y[:-10000]))
        ypred = clf.predict(self.train_x[-10000:])
        fpr, tpr, thresholds = metrics.roc_curve(self.train_y[-10000:], ypred, pos_label=1)
        for i in range(len(fpr)):
            plt.plot(fpr[i], tpr[i], "b*")
            plt.plot(fpr, tpr)
        plt.show()
        print "AUC : ", metrics.auc(fpr, tpr)
        print thresholds
        acc = 0.0
        for i in range(10000):
            if array(self.train_y)[len(self.train_y)-10000+i] == 1 and ypred[i] > 0.4:
                acc += 1
            if array(self.train_y)[len(self.train_y)-10000+i] == 0 and ypred[i] <= 0.4:
                acc += 1
        print "Accuracy : ", acc/10000


class Extremely_randomTree:                     # Extremely random tree

    def __init__(self, missing='mean'):
        self.rnd = 12
        random.seed(self.rnd)
        self.n_ft = 20 #Number of features to add
        self.max_elts = 3 #Maximum size of a group of linear features
        self.missing = missing

    def doclassify(self):
        train = loader("train.csv")
        target = train['target'].values
        test = loader("test.csv")
        id_test = test['ID'].values

        train['v22-1'] = train['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[0]))
        test['v22-1'] = test['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[0]))
        train['v22-2'] = train['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[1]))
        test['v22-2'] = test['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[1]))       # to process v22 features which is an important categorical feature
        train['v22-3'] = train['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[2]))
        test['v22-3'] = test['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[2]))
        train['v22-4'] = train['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[3]))
        test['v22-4'] = test['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[3]))

        drop_list = ['v91', 'v1', 'v8', 'v25', 'v29', 'v34', 'v41', 'v46', 'v54', 'v67', 'v97', 'v105', 'v122', 'v38', 'v72','v24', 'v52']
        train = train.drop(['ID', 'target'] + drop_list, axis=1).fillna(self.missing)
        test = test.drop(['ID'] + drop_list, axis=1).fillna(self.missing)
        # train, train_y = dp.Pandas_dataProcess().label_extract(train, real_extract=True)
        # test, id = dp.Pandas_dataProcess().label_extract(test, id=True, istest=True)
        # train, feature_name = dp.Pandas_dataProcess().trans_categorical(train)
        # test = dp.Pandas_dataProcess().trans_categorical(test, feature_name=feature_name, istest=True)
        # train = dp.Pandas_dataProcess().fill_missvalue(train, value='mean')
        # test = dp.Pandas_dataProcess().fill_missvalue(test, value='mean')
        refcols=list(train.columns)
        print refcols
        for elt in refcols:
            if train[elt].dtype == 'O':
                train[elt], temp = pd.factorize(train[elt])
                test[elt] = temp.get_indexer(test[elt])
            else:
                train[elt] = train[elt].round(5)
                test[elt] = test[elt].round(5)
        a = dp.AddNearestNeighbourLinearFeatures(n_neighbours=self.n_ft, max_elts=self.max_elts, verbose=True, random_state=self.rnd)
        a.fit(train, target)
        train = a.transform(train)
        test = a.transform(test)
        clf = ensemble.ExtraTreesClassifier(n_estimators=1200, max_features=30, criterion='entropy', min_samples_split=2,
                                max_depth=35, min_samples_leaf=2, n_jobs=-1, random_state=self.rnd)
        clf.fit(train,target)
        pred_et = clf.predict_proba(test)
        submission = pd.read_csv('sample_submission.csv')
        submission.index = submission.ID
        submission.PredictedProb = pred_et[:, 1]
        submission.to_csv('./addNNLinearFt.csv', index=False)
        submission.PredictedProb.hist(bins=30)
