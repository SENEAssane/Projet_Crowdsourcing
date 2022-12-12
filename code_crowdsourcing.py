# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:38:41 2022

@author: olivi
"""

# %%
import pandas as pd

df_cifar10h = pd.read_csv(
    'C:/Users/olivi/Documents/Montpellier/M2_SSD/Projet_Salmon/data/cifar10h-raw.csv', na_values='-99999')

df_cifar10h.dropna(inplace=True)
# %%
len(df_cifar10h)
len(df_cifar10h['image_filename'].unique())

import os
import numpy as np
cwd = os.getcwd()
path = cwd + '\\data'
array1 = np.load(path + '\\cifar10h-counts.npy')
array2 = np.load(path + '\\cifar10h-probs.npy')