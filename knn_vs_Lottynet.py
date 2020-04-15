# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:19:22 2020

@author: rfuchs
"""

import numpy as np


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
from sklearn.metrics import confusion_matrix


os.chdir('C:/Users/rfuchs/Documents/GitHub/planktonPipeline/extract_Pulse_values')

# Load nomenclature
tn = pd.read_csv('train_test_nomenclature.csv')
tn.columns = ['Particle_class', 'label']

###################################################################################################################
# Evaluate knn : Let the whole sample, should give an advantage 
###################################################################################################################

from viz_functions import plot_2D
from time import time
from scipy.integrate import trapz
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

os.chdir('C:/Users/rfuchs/Documents/cyto_classif')


#===========================================
# Without undersampling
#===========================================

X_train = np.load('FUMSECK_L3/X_train610.npy')
y_train = np.load('FUMSECK_L3/y_train610.npy')

X_valid = np.load('FUMSECK_L3/X_valid610.npy')
y_valid = np.load('FUMSECK_L3/y_valid610.npy')

X_test = np.load('FUMSECK_L3/X_test610.npy')
y_test = np.load('FUMSECK_L3/y_test610.npy')

# Integrate the curves
X_train_i = trapz(X_train, axis = 1)
X_valid_i = trapz(X_valid, axis = 1)
X_test_i = trapz(X_test, axis = 1)

knn_perfs = pd.DataFrame(columns = ['k', 'micro', 'macro', 'weighted'])

for k in range(1,10):
    print(k)
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train_i, y_train)
    y_pred_valid = knn.predict(X_valid_i)
    knn_perfs = knn_perfs.append({'k': k, 'micro': precision_score(y_valid, y_pred_valid, average = 'micro'), \
                    'macro': precision_score(y_valid, y_pred_valid, average='macro'), 
                    'weighted': precision_score(y_valid, y_pred_valid, average='weighted')}, 
                   ignore_index = True)

plt.plot(knn_perfs['k'], knn_perfs['micro'])
plt.plot(knn_perfs['k'], knn_perfs['macro'])
plt.plot(knn_perfs['k'], knn_perfs['weighted'])

# k = 2 seems to be best choice !


#===========================================
# With undersampling
#===========================================

X_train = np.load('FUMSECK_L3/X_train610.npy')
y_train = np.load('FUMSECK_L3/y_train610.npy')

X_valid = np.load('FUMSECK_L3/X_valid610.npy')
y_valid = np.load('FUMSECK_L3/y_valid610.npy')

X_test = np.load('FUMSECK_L3/X_test610.npy')
y_test = np.load('FUMSECK_L3/y_test610.npy')


X_integrated = trapz(X_train, axis = 1)
X_integrated = pd.DataFrame(X_integrated, columns = ['SWS','FWS', 'FL Orange', 'FL Red', 'Curvature'])
y = y_train.argmax(1)
  
# ENN for cleaning data
enn = EditedNearestNeighbours()
X_rs, y_rs = enn.fit_resample(X_integrated, y) 

X_train = X_train.take(enn.sample_indices_, axis = 0)
y_train = y_train.take(enn.sample_indices_, axis = 0)

# Rus to decrease sample size
balancing_dict = Counter(np.argmax(y_train,axis = 1))
for class_, obs_nb in balancing_dict.items():
    if obs_nb > 3000:
        balancing_dict[class_] = 3000


rus = RandomUnderSampler(sampling_strategy = balancing_dict)
ids = np.arange(len(X_train)).reshape((-1, 1))
ids_rs, y_train = rus.fit_sample(ids, y_train)
X_train = X_train[ids_rs.flatten()] 

# Integrate the curves
X_train_i = trapz(X_train, axis = 1)
X_valid_i = trapz(X_valid, axis = 1)
X_test_i = trapz(X_test, axis = 1)


knn_perfs = pd.DataFrame(columns = ['k', 'micro', 'macro', 'weighted'])

k = 2
for k in range(1,10):
    print(k)
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train_i, y_train)
    y_pred_valid = knn.predict(X_valid_i)
    knn_perfs = knn_perfs.append({'k': k, 'micro': precision_score(y_valid, y_pred_valid, average = 'micro'), \
                    'macro': precision_score(y_valid, y_pred_valid, average='macro'), 
                    'weighted': precision_score(y_valid, y_pred_valid, average='weighted')}, 
                   ignore_index = True)

plt.plot(knn_perfs['k'], knn_perfs['micro'])
plt.plot(knn_perfs['k'], knn_perfs['macro'])
plt.plot(knn_perfs['k'], knn_perfs['weighted'])

# k = 2 seems to be best choice ! (without ENN)


###################################################################################################################
# Evaluate ConvNet 
###################################################################################################################

from keras.models import load_model

# Performance of knn
# Load pre-trained model
LottyNet = load_model('C:/Users/rfuchs/Documents/cyto_classif/LottyNet_FUMSECK') 
y_pred_conv = LottyNet.predict(X_test)

precision_score(np.argmax(y_test, 1), np.argmax(y_pred_conv, 1), average = 'micro')
precision_score(np.argmax(y_test, 1), np.argmax(y_pred_conv, 1), average = 'macro')
precision_score(np.argmax(y_test, 1), np.argmax(y_pred_conv, 1), average = 'weighted')



###################################################################################################################
# Final word : Small win of ConvNet
###################################################################################################################

# Recap:
# Without undersampling
#           2-nn not us  2-nn rus   2-nn rus enn   2-nn enn       Convnet
# micro     0.884963     0.868984   0.844333        0.846938      0.937246
# macro     0.532623     0.500137   0.491953        0.499227      0.500073
# weighted  0.964046     0.954315   0.953897        0.955663      0.970292

# Gagne sur micro : + 5 points de pourcentage, perd de 3 sur macro et gagne de 0,6 sur weighted
# Certaine des petites classes sont un peu mieux représentées par knn sur L'ENSEMBLE DES DONNEES
# Quand réduit la taille du jeu de données, NN est vraiment gagnant


