##############################################################################
# TP2: Mon premier réseau de neurones
##############################################################################
# coding: utf-8

## TODO:
## tester le réseau "à la main", en présentant une image à la main
## ou en lui faisant écrire un tag sur les images, du style réponse du réseau


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import sys
import os
import pickle

# taille des images du jeu de données
img_h = 28
img_w = 28
input_shape = (img_h, img_w, 1)

train_data_nb = 10000
test_data_nb = 2000

print('- reading sample form the MNIST database')
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print('- keeping %d train samples and %d test samples' % (train_data_nb, test_data_nb))
x_train = x_train[0:train_data_nb,:,:]
y_train = y_train[0:train_data_nb]
x_test = x_test[0:test_data_nb,:,:]
y_test = y_test[0:test_data_nb]

def extract_samples(X,Y,y_subset):
    print('  - extracting samples of label ' + str(y_subset))
    N = Y.shape[0]
    XX = np.array([])
    YY = np.array([])
    for i in range(N):
        if Y[i] in y_subset:
            XX = np.append(XX, X[i])
            YY = np.append(YY, Y[i])
    XX = XX.reshape(-1,img_h,img_w)
    print('    %d samples of size %s' % (XX.shape[0], str(XX.shape)))
    return (XX,YY)

selected_y_values = [1,8]

print('- selecting training data:')
x_train, y_train = extract_samples(x_train, y_train, selected_y_values)
print('- selecting test data:')
x_test, y_test = extract_samples(x_test, y_test, selected_y_values)

print('- training data shape : ', x_train.shape)
print('- %d training samples' % x_train.shape[0])
print('- %d test samples' % x_test.shape[0])

print('- definition of the neural network')
# normalisation des valeurs
num_classes = 2
def renum(Y):
    for i in range(Y.shape[0]):
        for j in range(len(selected_y_values)):
            if Y[i] == selected_y_values[j]: Y[i] = j
renum(y_train)
renum(y_test)

# adaptation des formats de données pour les résultats
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

## QUESTION: Expliquez les 2 lignes de code suivantes:
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# les images doivent avoir pour dimension 28x28x1 (1 car c'est une image en N&B)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# si le modele a déjà été entrainé et enregistré, on le charge 
save_model = False # désactivation du mécanisme de sauvegarde
model_file = 'keras_model.h5'
history_file = 'keras_model_history.pickle'
if save_model and os.path.exists(model_file):
    print('- loading model from ')
    model = keras.models.load_model(model_file)
    history = pickle.load( open( history_file, 'rb' ) )
else:
    # définition de l'architecture du réseau de neurone
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(4, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.summary()

    # entrainement du réseau
    print('- training phase')
    batch_size = 128
    epochs = 8
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    H = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    
    # sauvegarde du réseau entrainé
    print('- saving the neural network into ' + model_file)
    model.save(model_file)
    pickle.dump(H.history, open(history_file, 'wb') )
    history = H.history

# evaluation du réseau sur le jeu de test
print('- evaluation of the neural network:')
score = model.evaluate(x_test, y_test)
print('  - test loss: ', score[0])
print('  - test accuracy:', score[1])

# Visualisation de la courbe d'apprentissage en fonction du nombre d'epoch
loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='training loss')
plt.plot(epochs, val_loss, 'r', label='validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

## QUESTION: Hormis l'architecture, donnez les principaux paramètres du réseau

## QUESTION: Pour chacun des paramètres suivants, établir un graphique des performances du réseau:
## - batch size
## - epochs
## - nombre de samples d'entrainement
## - durée de l'entrainement
## - nombre de filtres/features de la couche de convolution Conv2D
## - taille du noyau de la couche de convolution Conv2D

## QUESTION: Comparez les performances du réseaux en ajoutant une couche Conv2D/MaxPooling2D 
## juste après celle déjà existante dans l'architecture du réseau

##############################################################################
