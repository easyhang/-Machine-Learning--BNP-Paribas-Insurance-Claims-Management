'''Output data to current folder
Output data file are not available in other directory.
'''

import csv

def writer(id, ypred):                                  # write a submission based on particular type
    predictfile = open("BNP.csv", "w")
    predictfile_obj = csv.writer(predictfile)
    predictfile_obj.writerow(["ID", "PredictedProb"])
    for i in range(len(ypred)):
        predictfile_obj.writerow([id[i], ypred[i]])