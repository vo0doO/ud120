#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    """ Учитывая открытый файл электронной почты F, анализируйте весь текст ниже
        Метаданный блок на вершине
        (В части 2 вы также будете добавлять возможности stemming)
        и вернуть строку, которая содержит все слова
        В электронном письме (отделен для пробела)
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### Вернуться к началу файла (раздражает)
    all_text = f.read()

    ### Разделить метаданные
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### Удалить пунктуацию
        text_string = content[1].translate(str.maketrans("", "", string.punctuation))

        ### project part 2: Прокомментируйте линию ниже
        # words = text_string
        ### Разделите текстовую строку в отдельное слова, stem each word,
        ### and append the stemmed word to words (убедитесь, что есть один
        ### пространство между каждым stemmed word)

        stemmer = SnowballStemmer("english")
        words = ' '.join([stemmer.stem(word) for word in text_string.split()])




    return words
    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print(text)



if __name__ == '__main__':
    main()

