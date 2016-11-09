'''
    This part is to visualize the dataset, by using functions in this module, We will have the graph of relation of each
feature, the value distribution of each feature, relation of target and each feeture, the distribution of target as well
as weight of each feature.
'''


import pandas as pd
import time
import dataprocessing as dp
from numpy import *
from sklearn import datasets, linear_model
import collections
import dataloader
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau
from string import letters
import dataprocessing as dp

Not_number = ['v3', 'v22', 'v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110', 'v112', 'v113', 'v125']
class Paint():

    def missingVL_abs(self, dataset):                                                  #draw pictures of absolute number of missing values of each feature
        mpl.style.use('ggplot')
        # mpl.style.use('fivethirtyeight')
        dataset = dataset.drop(['ID', 'target'], axis=1)
        df = pd.isnull(dataset).astype(int)
        df=df[df.sum(axis=0).sort_values(axis=0, ascending=False).index]
        fig, ax = plt.subplots(figsize=(10.24, 7.68))
        ax.grid(False)
        bar = ax.bar(range(len(df.columns)),df.sum(axis=0).values)
        plt.xticks(0.5+arange(len(df.columns)), df.columns, fontsize=8, rotation=90)
        plt.axis('tight')
        plt.tight_layout()
        fname = 'figure/BNP_NA_Absolute_Num.png'
        plt.savefig(fname)


    def label_prop(self, dataset, feature):                                             #draw label distribution of labes over seperate values of each categorical feature

        p1 = dataset[['target', feature]]
        sns.set(style="whitegrid")

        g = sns.factorplot(x=str(feature), y="target", data=p1,
                   size=7, aspect=1.0, kind="bar", palette="muted")
        g.despine(left=True)
        g.set_ylabels("target")
        fname = 'figure/BNP_label_prop_' + feature + '.png'
        sns.plt.savefig(fname)
        sns.plt.show()

    def correlation_digonal(self, dataset):                                             #draw joint distribution of all pair of numerical features in dataset
        sns.set(style='white')
        sns.set_context("poster", font_scale=1)
        set_index = ['target']
        for i in range(131):
            feature = 'v'+str(i+1)
            if feature not in Not_number:
                set_index.append(feature)

        p1=dataset[['v99', 'v100', 'v101','v102','v103','v104','target']]
        p2= dataset[set_index]
        p3 = dataset[['v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']]

#         rs = random.RandomState(33)
#         d = pd.DataFrame(data=rs.normal(size=(100, 26)), columns=list(letters[:26]))
        corr = p3.corr()
        # corr = d.corr()
        mask = zeros_like(corr)
        mask[triu_indices_from(mask)] = True
        f, ax = plt.subplots(figsize=(12, 9))
        cmap = sns.diverging_palette(240, 5, as_cmap=True)
        # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, square=True, xticklabels=1, yticklabels=1, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        sns.plt.tight_layout()
        fname = 'figure/BNP_Corelation_sample.png'
        f_name = 'figure/BNP_Corelation_v32-28-18.png'
        sns.plt.savefig(fname)
        # sns.plt.show()

    def hexbin(self, dataset, x1, x2):                                               #draw joint distribution of two features
        sns.set(style="ticks")
        y = dataset[x1]
        x = dataset[x2]
        f, ax = plt.subplots(figsize=(19.2, 12.8))
        sns.jointplot(x, y, kind="hex", stat_func=kendalltau, color="#99004c")
        fname = 'figure/BNP_Kendall_'+x1+'-'+x2+'.png'
        sns.plt.savefig(fname)
        # sns.plt.show()

    def distribution(selfself, dataset, feature):                                   #draw distribution of single feature
        dm = dataset[[feature]]
        dm = dm.values.tolist()
        subdm = []
        for item in dm:
            if item[0] < inf:
                subdm.append(item[0])
        d = pd.DataFrame(subdm)
        sns.set(style="white", palette="muted", color_codes=True)
        f, axes = plt.subplots(figsize=(10.24, 7.68), sharex=True)
        sns.despine(left=True)
        # sns.distplot(d, hist=False, rug=True, color="r", ax=axes[0, 1])
        # sns.distplot(d, hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])
        sns.distplot(d, color="g", kde=False)
        # plt.legend()
        # plt.tight_layout()
        fname = 'figure/BNP_'+ feature +'_distribution.png'
        sns.plt.savefig(fname)
        # sns.plt.show()

    def paint_try(self):
        rs = random.RandomState(33)
        d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                 columns=list(letters[:26]))
        print rs
        print '``````````````````````````````````````````````````````'
        print d


def painting():
    Not_number = ['v3', 'v22', 'v24', 'v30', 'v31', 'v47', 'v52',
                  'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91',
                  'v107', 'v110', 'v112', 'v113', 'v125']
    start = time.time()
    print 'painting...'
    dataset = dataloader.loader('train.csv')
    p = Paint()
    # p.missingVL_abs(dataset)
    # p.correlation_digonal(dataset)
    # p.hexbin(dataset, 'v83', 'v130')
    # p.hexbin(dataset, 'v1', 'v37')
    p.label_prop(dataset, 'v125')
    # p.label_prop(dataset, 'v24')
    # p.hexbin(dataset, 'v10', 'v131')
    # p.distribution(dataset, 'v1')
    # p.distribution(dataset, 'v131')
    # p.distribution(dataset, 'target')
    end = time.time()
    print '-'*100
    print 'total run time is:', end-start, 's'

        








