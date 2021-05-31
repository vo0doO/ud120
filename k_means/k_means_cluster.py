#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ какой-то код построения, предназначенный для того, чтобы помочь вам визуализировать ваши кластеры """

    ### Участок каждый кластер с другим цветом - добавь больше цветов для
    ### Рисование более пяти кластеров
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### Если вам нравится, поместите красные звезды над точками, которые являются POIS (только для знакомых)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset_dos.pkl", "rb") )
### Есть выброс - удалить его!
data_dict.pop("TOTAL", 0)


### Функции ввода, которые мы хотим использовать
### Может быть любой ключ в словаре на уровне человека (заработная плата, режиссер_Fees и т. Д.)
feature_1 = "salary"
feature_2 = "exercised_stock_options"
# feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### В «кластеризации с 3 функциями» часть мини-проекта,
### Вы захотите изменить эту строку в
### for f1, f2, _ in finance_features:
### (Как сейчас написано, линия ниже предполагает 2 особенности)
for f1, f2 in finance_features:
    plt.scatter( f1, f2 )
plt.show()

### кластер здесь; Создать прогнозы кластерных меток
### Для данных и хранить их в список под названием PRE
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2).fit(finance_features)
pred = kmeans.predict(finance_features)



### Переименуйте параметр «Имя», когда вы измените количество функций
### так что цифра сохраняется в другой файл
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters2.pdf", f1_name=feature_1, f2_name=feature_2)
    stop = "s"
except NameError:
    print("no predictions object named pred found, no clusters to plot")
