#!/usr/bin/python

""" 
    Общий инструмент для преобразования данных из
    формат словаря в список Python (n x k), который
    готов для обучения алгоритму sklearn
    п - нет. пар ключ-значение в словаре
    к - нет. извлекаемых функций
    ключи словаря - это имена людей в наборе данных
    значения словаря - это словари, где каждый
        пара ключ-значение в dict - это имя
        функции и ее ценность для этого человека
    Помимо преобразования словаря в numpy
    массив, вы можете отделить метки от
    features - это то, для чего предназначен targetFeatureSplit
    Итак, если вы хотите использовать метку poi в качестве цели,
    а функции, которые вы хотите использовать, принадлежат человеку
    зарплата и бонус, вот что бы вы сделали:
    feature_list = ["poi", "зарплата", "бонус"]
    data_array = featureFormat (словарь_данных, список_список)
    метка, features = targetFeatureSplit (массив_данных)
    строка выше (targetFeatureSplit) предполагает, что
    label - это _первый_ элемент в feature_list - очень важно
    этот пои указан первым!
"""


import numpy as np

def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    """ преобразовать словарь в массив функций
        remove_NaN = True преобразует строку «NaN» в 0,0
        remove_all_zeroes = True будет опускать любые точки данных, для которых
            все функции, которые вы ищете, - 0,0
        remove_any_zeroes = True будет опускать любые точки данных, для которых
            любая из функций, которые вы ищете, - 0,0
        sort_keys = True сортирует ключи в алфавитном порядке. Установка значения как
            строка открывает соответствующий файл рассола с заданным ключом
            порядок (используется для совместимости с Python 3, а sort_keys
            следует оставить False для мини-проектов курса).
        ПРИМЕЧАНИЕ: первая функция считается poi и не проверяется на
            удаление нулевых или отсутствующих значений.
    """


    return_list = []

    # Порядок ключей - первая ветка предназначена для совместимости с Python 3 в мини-проектах,
    # вторая ветвь предназначена для совместимости в финальном проекте.
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print("error: key ", feature, " not present")
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        # Логика для принятия решения о добавлении точки данных.
        append = True
        # исключить класс poi в качестве критерия.
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### если все функции равны нулю и вы хотите удалить
        ### точки данных, которые все равны нулю, сделайте это здесь
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### если какие-либо функции для данной точки данных равны нулю
        ### и вы хотите удалить точки данных с любыми нулями,
        ### справиться с этим здесь
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Добавьте точку данных, если она отмечена для добавления.
        if append:
            return_list.append( np.array(tmp_list) )

    return np.array(return_list)


def targetFeatureSplit( data ):
    """
        учитывая массив numpy, подобный тому, который возвращается из
        featureFormat, выделите первую функцию
        и поместите его в отдельный список (это должен быть
        количество, которое вы хотите предсказать)
        возвращать цели и функции в виде отдельных списков
        (sklearn обычно может обрабатывать как списки, так и массивы numpy как
        форматы ввода при обучении / прогнозировании)
    """

    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    return target, features