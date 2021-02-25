#!/usr/bin/python

import random
import numpy
import matplotlib.pyplot as plt
import pickle

from outliers.outlier_cleaner import outlierCleaner


### загрузить некоторые данные практики с выбросами в нем
ages = pickle.load( open("practice_outliers_ages.pkl", "r") )
net_worths = pickle.load( open("practice_outliers_net_worths.pkl", "r") )



### age и net_worths необходимо преобразовать в двумерные массивы numpy
### второй аргумент команды reshape - кортеж целых чисел: (n_rows, n_columns)
### по соглашению n_rows - это количество точек данных
### и n_columns - количество функций
ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))
from sklearn.model_selection import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

### заполните регрессию здесь! Назовите объект регрессии reg, чтобы
### приведенный ниже код построения графика работает, и вы можете увидеть, как выглядит ваша регрессия











try:
    plt.plot(ages, reg.predict(ages), color="blue")
except NameError:
    pass
plt.scatter(ages, net_worths)
plt.show()


### выявить и удалить наиболее резко выделяющиеся точки
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    cleaned_data = outlierCleaner( predictions, ages_train, net_worths_train )
except NameError:
    print("ваш объект регрессии не существует или не имеет имени reg")
    print("не может делать прогнозы для использования при выявлении выбросов")

### запускайте этот код только в том случае, если cleaned_data возвращает данные
if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

    ### восстановить ваши очищенные данные!
    try:
        reg.fit(ages, net_worths)
        plt.plot(ages, reg.predict(ages), color="blue")
    except NameError:
        print("похоже, что регрессия не импортирована / не создана,")
        print("иначе ваш объект регрессии не назван reg")
        print("в любом случае, нарисуйте только диаграмму рассеяния очищенных данных")
    plt.scatter(ages, net_worths)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()


else:
    print("outlierCleaner() возвращает пустой список, переустановка не требуется")

