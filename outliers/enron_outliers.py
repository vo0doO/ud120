#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from tools.feature_format import featureFormat, targetFeatureSplit


### прочитать в словаре данных, преобразовать в массив numpy
data_dict = pickle.load( open("../final_project/final_project_dataset_dos.pkl", "r+b") )

data_dict.pop("TOTAL", 0)

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### ваш код ниже
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus)
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

stop = "stop"