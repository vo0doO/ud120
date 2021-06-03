#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append("../tools/")
from parse_out_email_text import parseOutText

"""
    Стартовый код для обработки электронных писем от Sara и Chris для извлечения
    Особенности и получают документы, готовые к классификации.

    Список всех электронных писем из SARA находятся в списке из_сара
    Аналогично для электронных писем из Криса (from_chris)

    Фактические документы находятся в наборе электронного письма Enron, который
    Вы загрузили / распасываете частью 0 первого мини-проекта. Если у тебя есть
    Не получили Enron Email Corpus, запустите Startup.py в папке «Инструменты».

    Данные хранятся в списках и упакованы в расчетные файлы в конце.
"""

from_sara = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter - это способ ускорить развитие - есть
### тысячи электронных писем из Сары и Криса, так бегают над всеми из них
### может занять много времени
### temp_counter поможет вам только смотреть на первые 200 электронных писем в списке, чтобы вы
### Можно повторить свои модификации быстрее
temp_counter = 0

for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### только посмотреть на первые 200 электронных писем при разработке
        ### После того, как все работает, удалите эту строку, чтобы запустить полный набор данных
        path = os.path.join('..', path[:-1])
        print(path)
        email = open(path, "r")

        ### Используйте ParseOutText, чтобы извлечь текст из открытого электронного письма
        text_string = parseOutText(email)
        ### использовать str.replace() Удалить любые случаи слов
        rep_list = ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf"]
        for word in rep_list:
            text_string = text_string.replace(word, "")

        ### Добавьте текст в Word_Data
        word_data.append(text_string)
        ### добавить а 0 to from_data Если электронная почта от Sara, а 1, если электронная почта от Криса
        from_data.append(0 if name == "sara" else 1)

        email.close()

print("emails processed")
from_sara.close()
from_chris.close()

pickle.dump(word_data, open("your_word_data.pkl", "wb"))
pickle.dump(from_data, open("your_email_authors.pkl", "wb"))

### in Part 4, do TfIdf vectorization here
from nltk.corpus import stopwords
sw = stopwords.words("english")
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english", lowercase=True)
bag_of_words = vectorizer.fit_transform(word_data)
print(len(vectorizer.get_feature_names()))
