##############################################################################
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import sys
import os
import pandas

# taille des images du jeu de données
img_h = 28
img_w = 28
input_shape = (img_h, img_w, 1)

train_data_nb = 10000
test_data_nb = 2000

print('- reading sample form the MNIST database')
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

def draw_example(X,Y):
    fig, ax = plt.subplots(6, 6, figsize = (12, 12))
    fig.suptitle('First 36 images in MNIST')
    fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
    for x, y in [(i, j) for i in range(6) for j in range(6)]:
        ax[x, y].imshow(X[x + y * 6].reshape((28, 28)), cmap = 'gray')
        ax[x, y].set_title(Y[x + y * 6])
    plt.show()

print('- drawing samples')
draw_example(x_train, y_train)

## QUESTION: selectionnez les train_data_nb premiers samples d'entrainement chargés, 
## idem pour les samples de test

## QUESTION: Pour cette première approche, on va essayer de discriminer deux chiffres
## seulement: Selectionnez les data représentant seulement les chiffres 1 et 8
## et remplacez les tableaux x_train, y_train, x_test, y_test avec ces nouvelles données

print('- training data shape : ', x_train.shape)
print('- %d training samples' % x_train.shape[0])
print('- %d test samples' % x_test.shape[0])

print('- drawing samples')
draw_example(x_train, y_train)

## QUESTION: Calculez une colonne n_train indiquant la proportion de pixels non noir 
## pour chaque sample

## QUESTION: Elaborez un diagramme en baton représentant les proportions des pixels
## blancs pour chacunes des valeurs étudiées (sur un unique diagramme) 

## QUESTION: Donnez un seuil pour discriminer les valeurs 1 et 8

## QUESTION: Testez ce seuil pour les valeurs d'entrainement, 
## donnez la proportion d'erreur de classification

## QUESTION: Calculez la moyenne et l'écart type de la proportion 
## de pixels blancs pour chaque valeur considérée

## QUESTION: En vous servant des calculs de moyennes et d'écart-type précédent, 
## et en considérant que la distribution suit une loi normale, donnez une 
## probabilité d'erreur pour le seuil précédent.

## QUESTION: Essayez de trouver d'autres critères (projection latérale/verticale, barycentre des pixels, transformation: erosion/dilatation, etc..)




