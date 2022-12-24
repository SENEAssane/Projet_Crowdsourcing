# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 08:46:54 2022

@author: olivi
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

df_cifar10h = pd.read_csv(
    'C:/Users/olivi/Documents/Montpellier/M2_SSD/Projet_Salmon/data/cifar10h-raw.csv', na_values='-99999')

df_cifar10h.dropna(inplace=True)
# df_category_label = df_cifar10h[["annotator_id","true_category","chosen_category",
#                                  "true_label", "chosen_label"]]


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
    df_labels = sub_df[['true_label', 'chosen_label']]
    return df_labels

df_labels = CreateSubDf(df_cifar10h, 4)

# Fonction affichant la matrice de confusion associée à un annotateur
# elle prend en argument:
# y_true tableau dans lequel sont stockés les vraies label
# y_pred tableau dans lequel sont stockés les labels choisis par l'annotateur
# Elle retourne un graphique représentant la matrice de confusion

def PlotConfusionMatrix(y_true, y_pred):
    arr_true_labels = df_labels[[y_true]].to_numpy()
    arr_chosen_labels = df_labels[[y_pred]].to_numpy()
    ConfusionMatrixDisplay.from_predictions(arr_true_labels, arr_chosen_labels)
    plt.show()

PlotConfusionMatrix('true_label', 'chosen_label')

# confusions = {}
# confusions[0] = confusion_matrix(df_labels['true_label'],
#                                  df_labels['chosen_label'],normalize='true')
