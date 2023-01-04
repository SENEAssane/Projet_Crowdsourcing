# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 08:46:54 2022

@author: olivi
"""

import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from isar.models import DawidSkeneIID


# Cette fonction créé un nouveau dataframe qui contiendra les labels choisis 
# par un annotateur specifique (choisi en argument) 
# Il contiendra également les vrais labels des images qui ont été anotées par
# cet individu
# Elle prend en argument:
# df le dataframe à partir duquel on va créer le sous dataframe
# annotator_id qui est un entier unique associé
# à l'annotateur
# Elle renvoie un dataframe composé de 2 colonnes 
# la colonne des vrais labels et celle des labels choisis par l'annotateur

def CreateSubDf(df, an_id):
    sub_df = df[(df["annotator_id"] == an_id)]
    df_labels = sub_df[['cifar10_test_test_idx', 'true_label', 'chosen_label']]
    return df_labels


# customisation de la matrice de confusion
# Cette fonction prend en argument:
# cm (la matrice de confusion)
# classes (la liste des noms de chaque classe)
# normalize (pour normaliser la matrice ou non)
# title (le titre de la représentation graphique de la matrice de confusion)
# cmap (palette de couleurs pour le graphique)
def custom_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True labels')
    plt.xlabel('Chosen labels')
    plt.tight_layout()
    

# Calcul de la matrice de contingence (ou matrice de confusion)
# Cette fonction prend en argument:
# y_true (un numpy array dans lequel sont stockés les vrais labels)
# y_predict (un numpy array dans lequel sont stockés les labels choisis par
# l'annotateur)
# class_names (la liste des noms de chaque classe)
# elle retourne le graphique customisé de la matrice de confusion
def plot_confusion_matrix(y_true, y_predict, class_names):
    conf_matrix = confusion_matrix(y_true, y_predict)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(15,15))
    custom_confusion_matrix(conf_matrix, normalize=True, classes=class_names,
                          title='Matrice de confusion')
    plt.show()


# compter le nombre de fois que le label chat est correctement prédit par l'annotateur
# df_labels[(df_labels['true_label'] == 3) & (df_labels['chosen_label'] == 9)].value_counts()
# compter le nombre de fois que l'annoteur se trompe en ne prédisant pas le label chat
# df_labels[(df_labels['true_label'] == 3) & (df_labels['chosen_label'] != 3)].value_counts()

def plot_cm_ds(df, an_id, labels):
    sub_df = CreateSubDf(df, an_id)
    U_df = sub_df['chosen_label'].values.astype(float)
    U2D_array = np.reshape(U_df, (-1, 1))
    pi = np.random.dirichlet(np.ones(10))
    psi = np.tile(np.eye(10)[:, np.newaxis, :]*0.93, [1, 1, 1]) + 0.01
    dsmodel = DawidSkeneIID((10, 1), predict_tol=0.5)
    dsmodel.fit(U2D_array, priors=(np.ones(10), np.ones([10, 1, 10])),
    starts=[(pi, psi)])
    cm3D_array = dsmodel.Psi
    cm = cm3D_array.reshape(10,10)
    plt.figure(figsize=(15,15))
    custom_confusion_matrix(
        cm, normalize=True, classes=labels,
        title="Matrice de confusion estimée avec l'algo EM")
    plt.show()
    


if __name__ == '__main__':
    df_cifar10h = pd.read_csv(
        'C:/Users/olivi/Documents/Montpellier/M2_SSD/Projet_Salmon_test/data/cifar10h-raw.csv',
        na_values='-99999')

    df_cifar10h.dropna(inplace=True)
    
    # df_labels = CreateSubDf(df_cifar10h, 10)
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
                "horse","ship", "truck"]
    
    annotator_id = 7
    plot_cm_ds(df_cifar10h, annotator_id, labels)
    
    y_pred_an = CreateSubDf(df_cifar10h, annotator_id)['chosen_label'].to_numpy()
    y_true = CreateSubDf(df_cifar10h, annotator_id)['true_label'].to_numpy()
    plot_confusion_matrix(y_true, y_pred_an, labels)