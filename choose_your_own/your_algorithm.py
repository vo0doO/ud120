#!/usr/bin/python

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from choose_your_own.prep_terrain_data import makeTerrainData
from choose_your_own.class_vis import prettyPicture
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import matplotlib.pyplot as plt
import gc
from datetime import datetime

# to store the results
scikit_classifier_results = []
scikit_regressor_results = []

mu_second = 0.0 + 10 ** 6  # number of microseconds in a second


def datly():

    # import
    features_train, labels_train, features_test, labels_test = makeTerrainData()

    def plotly(grade_fast, bumpy_fast, grade_slow, bumpy_slow):
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
        plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
        plt.legend()
        plt.xlabel("bumpiness")
        plt.ylabel("grade")
        plt.show()

    def composite():
        # обучающие данные (features_train, labels_train)
        # имеют как "быстрый", так и " медленный" точки смешиваются
        # вместе-разделите их, чтобы мы могли дать им разные цвета
        # в диаграмме рассеяния и определить их визуально
        grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
        bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
        grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
        bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

        plt = plotly(grade_fast, bumpy_fast, grade_slow, bumpy_slow)


    return features_train, labels_train, features_test, labels_test, plt


def plotly(clf):
    try:
        prettyPicture(clf, features_test, labels_test)
    except NameError:
        pass


def learnly():
    clf = ExtraTreesClassifier(n_estimators=30)
    clf.fit(features_train, labels_train)
    clf.predict(features_train)
    score = clf.score(features_test, labels_test)
    print(score)
    stop = "stop"
    return clf, score


def bench_scikit_tree_classifier(X, Y):
    """Benchmark with scikit-learn decision tree classifier"""

    from sklearn.tree import DecisionTreeClassifier

    gc.collect()

    # start time
    tstart = datetime.now()
    clf = DecisionTreeClassifier()
    clf.fit(X, Y)
    clf.predict(X)
    delta = (datetime.now() - tstart)
    # stop time

    scikit_classifier_results.append(
        delta.seconds + delta.microseconds / mu_second)


def bench_scikit_tree_regressor(X, Y):
    """Benchmark with scikit-learn decision tree regressor"""

    from sklearn.tree import DecisionTreeRegressor

    gc.collect()

    # start time
    tstart = datetime.now()
    clf = DecisionTreeRegressor()
    clf.fit(X, Y).predict(X)
    delta = (datetime.now() - tstart)
    # stop time

    scikit_regressor_results.append(
        delta.seconds + delta.microseconds / mu_second)

if __name__ == '__main__':
    # import
    import sys

    features_train, labels_train, features_test, labels_test, plt = datly()
    clf, score = learnly()
    plotly(clf=clf)
    X = features_train
    Y = labels_train

    # bench ?
    if sys.argv[0] == "bench":
        print('============================================')
        print('Warning: this is going to take a looong time')
        print('============================================')


        n = int(len(features_train))
        step = 1
        n_samples = int(len(labels_train))
        # dim = 10
        # n_classes = int(len(labels_train))
        for i in range(n):
            print('============================================')
            print('Entering iteration %s of %s' % (i, n))
            print('============================================')
            # n_samples += step
            # X = np.random.randn(n_samples, dim)
            # Y = np.random.randint(0, n_classes, (n_samples,))
            bench_scikit_tree_classifier(X, Y)
            # Y = np.random.randn(n_samples)
            bench_scikit_tree_regressor(X, Y)

        xx = range(0, n * step, step)
        plt.figure('scikit-learn tree benchmark results')
        plt.subplot(211)
        plt.title('Learning with varying number of samples')
        plt.plot(xx, scikit_classifier_results, 'g-', label='classification')
        plt.plot(xx, scikit_regressor_results, 'r-', label='regression')
        plt.legend(loc='upper left')
        plt.xlabel('number of samples')
        plt.ylabel('Time (s)')

        scikit_classifier_results = []
        scikit_regressor_results = []
        # n = 10
        step = 1
        start_dim = 0
        # n_classes = 10

        dim = start_dim
        for i in range(0, n):
            print('============================================')
            print('Entering iteration %s of %s' % (i, n))
            print('============================================')
            dim += step
            # X = np.random.randn(100, dim)
            # Y = np.random.randint(0, n_classes, (100,))
            bench_scikit_tree_classifier(X, Y)
            # Y = np.random.randn(100)
            bench_scikit_tree_regressor(X, Y)

        xx = np.arange(start_dim, start_dim + n * step, step)
        plt.subplot(212)
        plt.title('Learning in high dimensional spaces')
        plt.plot(xx, scikit_classifier_results, 'g-', label='classification')
        plt.plot(xx, scikit_regressor_results, 'r-', label='regression')
        plt.legend(loc='upper left')
        plt.xlabel('number of dimensions')
        plt.ylabel('Time (s)')
        plt.axis('tight')
        plt.show()
    # learn
    features_train, labels_train, features_test, labels_test, plt = datly()
    clf, score = learnly()
    plotly(clf=clf)

# ваш код!  назовите свой объект классификатора clf, если вы хотите
# код визуализации (prettyPicture), чтобы показать вам границу решения
