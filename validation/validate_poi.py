#!/usr/bin/python


"""
    Стартовый код для проверки мини-проекта.
    Первый шаг к созданию вашего идентификатора POI!

    Начните с загрузки / форматирования данных

    После этого, это больше не наш код-это ваш!
"""

import pickle
import sys

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset_dos.pkl", "rb"))

### первый элемент-это наши метки, любые добавленные элементы являются предиктором
### особенности. Держите это то же самое для мини-проекта, но вы будете
### есть другой список функций, когда вы делаете окончательный проект.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson13_keys_dos.pkl')
labels, features = targetFeatureSplit(data)

### с этого момента все твое!


s = "s"