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

data = featureFormat(data_dict, features_list, sort_keys='../tools/python2_lesson14_keys_dos.pkl')
labels, features = targetFeatureSplit(data)

### с этого момента все твое!
from sklearn import tree
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,
                                                                            random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)


def my_metrics(pr, lt):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    precision = precision_score(lt, pr)
    recall = recall_score(lt, pr)
    accuracy = accuracy_score(pr, lt)
    f1 = f1_score(pr, lt)

    print(
        {"accuracy": round(accuracy, 3)},
        {"precision": round(precision, 3)},
        {"recall": round(recall, 3)},
        {"f1_score": round(f1, 3)},
    )


predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

my_metrics(predictions, true_labels)

s = "s"
