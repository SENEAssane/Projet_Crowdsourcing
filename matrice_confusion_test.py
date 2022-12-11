# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 20:13:42 2022
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

dataset = load_digits()
X = dataset.data # Entrees
Y = dataset.target # Resultats attendus
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

activ = ['identity', 'logistic', 'tanh', 'relu']
solve = ['lbfgs', 'sgd', 'adam']
layers = [(),(7),(4,2),(3,3,2)]
index = []

for j in range(len(solve)) :
    for i in range(len(layers)) :
        index.append((solve[j],layers[i]))

a = []
M = pd.DataFrame(columns = activ)

for j in range(len(activ)) :
    for i in range(len(solve)) : 
        for k in range(len(layers)) :
            classifier = MLPClassifier(hidden_layer_sizes=layers[k],
                                      activation=activ[j],
                                      solver=solve[i])
            classifier.fit(X_train,Y_train)
            Y_pred = classifier.predict(X_test)
            a.append(accuracy_score(Y_test, Y_pred))     
    M.iloc[:,j] = a
    a = []

M.columns = activ
M.index = index

cm = confusion_matrix(Y_test, Y_pred, labels=classifier.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classifier.classes_)
disp.plot()
plt.show()