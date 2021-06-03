#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### слова (особенности) и авторы (этикетки), уже в значительной степени обработаны.
### Эти файлы должны были быть созданы из предыдущего (урок 10)
### мини-проект.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load(open(words_file, "rb"))
authors = pickle.load(open(authors_file, "rb"))



### test_size - это процент событий, назначенных на набор тестов (
### остаток идет на тренировку)
### Функциональные матрицы изменены на густые представления для совместимости с
### Классификатор функции в версиях 0.15.2 и ранее
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test).toarray()


### Классический способ переоснащения - использовать небольшое количество
### точек данных и большое количество функций;
### поезд на 150 событий, чтобы поставить себя в этот режим
features_train = features_train[:150].toarray()
labels_train = labels_train[:150]



### Ваш код идет здесь
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score

acc = accuracy_score(pred, labels_test)

def submitAccuracies():
    return {"acc": round(acc, 3)}
print(submitAccuracies())
for e in clf.feature_importances_:
    if e > 0.2:
        print(e)
        print(list(clf.feature_importances_).index(e))
        print(vectorizer.get_feature_names()[list(clf.feature_importances_).index(e)])


