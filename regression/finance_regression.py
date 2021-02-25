#!/usr/bin/python

"""
    Стартовый код для мини-проекта регрессии.

    Загружает / форматирует измененную версию набора данных
    (зачем модифицировать? мы удалили некоторые проблемы
    что вы попадете в мини-проект выбросов).

    Рисует небольшую диаграмму рассеяния данных обучения / тестирования

    Вы вводите код регрессии в указанном месте:
"""    


import sys
import pickle
sys.path.append("../tools/")
from tools.feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load( open("../final_project/final_project_dataset_modified_dos.pkl", "rb") )

### перечислите функции, на которые вы хотите обратить внимание - первый элемент в
### список будет "целевой" функцией
features_list = ["bonus", "salary"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True, sort_keys = '../tools/python2_lesson06_keys_dos.pkl')
target, features = targetFeatureSplit( data )

### разделение обучения и тестирования необходимо в регрессии, как и при классификации
from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"



### Ваш регресс идет сюда!
### Пожалуйста, назовите его reg, чтобы приведенный ниже код построения подобрал его и
### правильно рисует. Не забудьте изменить test_color выше с "b" на
### «r», чтобы отличать тренировочные точки от контрольных.
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(feature_train, target_train)




### нарисуйте диаграмму рассеяния с цветными обозначениями точек обучения и тестирования
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### этикетки для легенды
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")




### нарисуйте линию регрессии, как только она закодирована
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass
reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="b")
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
stop = "stop"