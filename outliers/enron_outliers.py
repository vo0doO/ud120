#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from tools.feature_format import featureFormat, targetFeatureSplit


### прочитать в словаре данных, преобразовать в массив numpy
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### ваш код ниже



