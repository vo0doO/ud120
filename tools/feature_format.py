#!/usr/bin/python

""" 
    Общий инструмент для преобразования данных из
    формат словаря в (n x k) список Python, это
    готов к обучению алгоритм склеарн

    n--no. пар ключ-значение в диктонарном
    k--no. извлекаемых объектов

    словарные ключи - это имена людей в наборе данных
    значения словаря являются словарями, где каждый
        пара ключ-значение в dict это имя
        функции, и ее значение для этого человека

    В дополнение к преобразованию словаря в NumPy
    массив, Вы можете отделить ярлыки от
    features--this для чего предназначен targetFeatureSplit

    так что, если вы хотите иметь poi label как цель,
    и features вы хотите использовать это person's
    salary and bonus, вот что вы могли бы сделать:

    feature_list = ["poi", "salary", "bonus"] 
    data_array = featureFormat( data_dictionary, feature_list )
    label, features = targetFeatureSplit(data_array)

    линия выше(targetFeatureSplit) предполагает, что
    label это _first_ item в feature_list--very важный
    тот poi в списке первый!
"""

import numpy as np


def featureFormat(dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False,
                  sort_keys=False):
    """ преобразовать словарь в массив функций
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True пропустит любые точки данных, для которых
            все функции, которые вы ищете 0.0
        remove_any_zeroes = True пропустит любые точки данных, для которых
            Любая из функций, которые вы ищете 0.0
        sort_keys = True сортирует ключи по алфавиту. Установка значения как
            строка открывает соответствующий файл выбора с заданным ключом
            порядок (используется для совместимости с Python 3 и sort_keys
            следует оставить как False для курса мини-проектов).
        NOTE: Предполагается, что первая особенность «poi» и не проверяется на
            удаление для нулевых или отсутствующих значений.
    """

    return_list = []

    # Key order - первая ветка для совместимости с Python 3 на мини-проектах,
    # вторая ветка для совместимости на финальном проекте.
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
            if value == "NaN" and remove_NaN:
                value = 0
            tmp_list.append(float(value))

        # Логика для принятия решения, добавлять или нет точку данных.
        append = True
        # исключить класс 'poi' в качестве критерия.
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
        ### Добавить точку данных, если помечено для добавления.
        if append:
            return_list.append(np.array(tmp_list))

    return np.array(return_list)


def targetFeatureSplit(data):
    """ 
        учитывая массив Numpy, как тот, который вернулся из
        FeatureFormat, выделите первую функцию
        и положить его в свой список (это должно быть
        количество, которое вы хотите предсказать)

        возвращать цели и функции в виде отдельных списков

        (sklearn как правило, может обрабатывать как списки, так и массивы
        форматы ввода при training/predicting)
    """

    target = []
    features = []
    for item in data:
        target.append(item[0])
        features.append(item[1:])

    return target, features
