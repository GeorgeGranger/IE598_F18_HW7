#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 17:21:29 2018

@author: huangsida
"""
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/'
                      'wine/wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol',
                   'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols',
                   'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315','Proline']
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    random_state=42,
                                                    stratify=y)

forest = RandomForestClassifier(criterion='gini',
                                n_estimators=90,
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)


scores_insample = cross_val_score(estimator=forest,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)

y_pred = forest.predict(X_test)

scores_outofsample = accuracy_score(y_test,y_pred)

print('In_sample CV accuracy scores: %s' % scores_insample)
a=np.mean(scores_insample)
print('Mean of In_sample CV accuracy scores: %s' % a)
print('Out-of-sample CV accuracy scores: %s\n' % scores_outofsample)

feat_labels = df_wine.columns[1:]
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
         feat_labels[indices[f]],
         importances[indices[f]]))
