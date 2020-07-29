# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:19:22 2020

@author: rfuchs
"""

import re
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_score, recall_score
from imblearn.under_sampling import EditedNearestNeighbours

os.chdir('C:/Users/rfuchs/Documents/GitHub/phyto_curves_reco')

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


def prec_rec_function(y_test, preds, cluster_classes, algo):
    ''' Compute the precision and recall for all classes'''
    prec = precision_score(y_test, preds, average=None)
    prec = dict(zip(cluster_classes, prec))
    prec['algorithm'] = 'knn'
    
    recall= recall_score(y_test, preds, average=None)
    recall = dict(zip(cluster_classes, recall))
    recall['algorithm'] = algo

    return prec, recall

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



############################################################
# Training of other algorithms on the unbiased dataset
############################################################
import fastparquet as fp

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler


from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import OneHotEncoder


from keras.utils import to_categorical
y_oh= to_categorical(y_train)


cluster_classes = ['airbubble', 'cryptophyte', 'nanoeucaryote',\
                   'inf1microm_unidentified_particle', 'microphytoplancton',\
                'picoeucaryote', 'prochlorococcus', \
                'sup1microm_unidentified_particle', 'synechococcus']

    
#************************************
# Data importation 
#************************************

# TRAIN/TEST SPLIT TO CHANGE 

list_dir = 'C:/Users/rfuchs/Documents/cyto_classif/XP_Listmodes'
list_files = os.listdir(list_dir)
parts_to_keep_files = [file for file in list_files if re.search('.csv', file)]
list_files = [file for file in list_files if re.search('.parq', file)]

# Import the list of the selected particles
pids_df = pd.DataFrame()
for file in parts_to_keep_files:
    df = pd.read_csv(list_dir + '/' + file)
    pids_df = pids_df.append(df)

#==============================================
# Build the train and test set
#==============================================

# Train set 
train_particles = pd.read_csv(list_dir + '/' + 'train_pids.csv')
train_files = np.unique(train_particles['acq'])
valid_particles = pd.read_csv(list_dir + '/' + 'valid_pids.csv')
valid_files = np.unique(valid_particles['acq'])

train = pd.DataFrame()

for file in np.concatenate([train_files, valid_files]):
    pf = fp.ParquetFile(list_dir + '/' + file)
    all_particles = pf.to_pandas()
    
    pids = list(pids_df[pids_df['acq'] == file]['Particle ID'])
    selected_particles = all_particles.set_index('Particle ID').loc[pids]
    
    train = train.append(selected_particles)

# Valid set 
test_particles = pd.read_csv(list_dir + '/' + 'test_pids.csv')
test_files = np.unique(test_particles['acq'])

test = pd.DataFrame()

for file in test_files:
    pf = fp.ParquetFile(list_dir + '/' + file)
    all_particles = pf.to_pandas()
    
    pids = list(pids_df[pids_df['acq'] == file]['Particle ID'])
    selected_particles = all_particles.set_index('Particle ID').loc[pids]
    
    test = test.append(selected_particles)

#
X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]

X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]


# Delete empty columns
X_train = X_train.iloc[:, X_train.columns != 'Curvature Center of gravity']
X_train = X_train.iloc[:, X_train.columns != 'Curvature Asymmetry']

X_test = X_test.iloc[:, X_test.columns != 'Curvature Center of gravity']
X_test = X_test.iloc[:, X_test.columns != 'Curvature Asymmetry']

p = X_train.shape[1]


# RUS for cleaning data (will be changed)
rus = RandomUnderSampler()
X_train, y_train = rus.fit_resample(X_train, y_train) 

np.unique(y_test, return_counts = True)

    
#************************************
# Looking for the best hyperparams 
#************************************
    



#********************************
# Fitting of the models
#********************************

# KNN
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, y_train)

# SVM
svm = svm.SVC()
svm.fit(X_train, y_train)

# LGBM
lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)

# FFNN
enc = OneHotEncoder(handle_unknown='ignore')
y_train_oh = enc.fit_transform(y_train.values.reshape(-1, 1))

# define the keras model
ffnn = Sequential()
ffnn.add(Dense(32, input_dim = p, activation='relu'))
ffnn.add(Dense(16, activation='relu'))
ffnn.add(Dense(9, activation='sigmoid'))
# compile the keras ffnn
ffnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras ffnn on the dataset
ffnn.fit(X_train, y_train_oh, epochs=10, batch_size=256)


# Add LottyNet !

#********************************
# Prediction of the models
#********************************

knn_preds = knn.predict(X_test)  
svm_preds = svm.predict(X_test)  
lgbm_preds = lgbm.predict(X_test) 
ffnn_preds = ffnn.predict(X_test) 
ffnn_preds = enc.inverse_transform(ffnn_preds)


#********************************
# Accuracy computations
#********************************

prec = pd.DataFrame(columns= cluster_classes + ['algorithm'])
recall = pd.DataFrame(columns= cluster_classes + ['algorithm'])


# KNN 
prec_knn, recall_knn = prec_rec_function(y_test, knn_preds, cluster_classes, 'knn')
prec = prec.append(prec_knn, ignore_index = True)
recall = recall.append(recall_knn, ignore_index = True)

# SVM
prec_svm, recall_svm = prec_rec_function(y_test, svm_preds, cluster_classes, 'svm')
prec = prec.append(prec_svm, ignore_index = True)
recall = recall.append(recall_svm, ignore_index = True)


# LGBM
prec_lgbm, recall_lgbm = prec_rec_function(y_test, lgbm_preds, cluster_classes, 'lgbm')
prec = prec.append(prec_lgbm, ignore_index = True)
recall = recall.append(recall_lgbm, ignore_index = True)

# FFNN
prec_ffnn, recall_ffnn = prec_rec_function(y_test, ffnn_preds, cluster_classes, 'ffnn')
prec = prec.append(prec_ffnn, ignore_index = True)
recall = recall.append(recall_ffnn, ignore_index = True)

#********************************
# Final output
#********************************

prec_recall_comp = pd.concat([prec, recall], axis = 1)
prec_recall_comp.columns = [cc + '_prec' for cc in cluster_classes] + \
                ['algorithm_prec'] + [cc + '_recall' for cc in cluster_classes] +\
                    ['algorithm']

prec_recall_comp = prec_recall_comp.drop('algorithm_prec', 1)

prec_recall_comp.set_index('algorithm').to_csv('prec_recall_comp.csv')
