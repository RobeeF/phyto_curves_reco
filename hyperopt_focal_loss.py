# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 13:48:42 2020

@author: rfuchs
"""


from __future__ import print_function
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe

from hyperas import optim
from hyperas.distributions import choice, uniform

import os
from collections import Counter
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from keras.layers import Input, Conv1D,  GlobalAveragePooling1D, Dense, Dropout
from keras.models import Model
from keras import metrics

from keras import optimizers

from sklearn.metrics import confusion_matrix, precision_score
from losses import categorical_focal_loss

##############################################################################################
#################  Model 13 Hyper-parameters tuning on FUMSECK Data ##########################
##############################################################################################

# Tuning include : Sampling strategy (RUS vs RUS + ENN)

def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """

    train = np.load('train.npz')
    valid = np.load('valid.npz')
    test = np.load('valid.npz')
    
    X_train = train['X']
    X_valid = valid['X']
    X_test = test['X']
    
    y_train = train['y']
    y_valid = valid['y']
    y_test = test['y']
    
    tn = pd.read_csv('train_test_nomenclature.csv')
    tn.columns = ['Particle_class', 'label']  
            
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def create_model(X_train, y_train, X_valid, y_valid, X_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """


    dp = {{uniform(0, 0.5)}}
    
    N_CLASSES = y_train.shape[1]
    max_len = X_train.shape[1]
    nb_curves = X_train.shape[2]
    
    sequence_input = tf.keras.layers.Input(shape=(max_len, nb_curves), dtype='float32')
    
    # A 1D convolution with 128 output channels: Extract features from the curves
    x = tf.keras.layers.Conv1D(64, 5, activation='relu')(sequence_input)
    x = tf.keras.layers.Conv1D(32, 5, activation='relu')(x)
    x = tf.keras.layers.Conv1D(16, 5, activation='relu')(x)

    # Average those features
    average = tf.keras.layers.GlobalAveragePooling1D()(x)
    dense2 = tf.keras.layers.Dense(32, activation='relu')(average) # Does using 2*32 layers make sense ?
    drop2 = tf.keras.layers.Dropout(dp)(dense2)
    dense3 = tf.keras.layers.Dense(32, activation='relu')(drop2)
    drop3 = tf.keras.layers.Dropout(dp)(dense3)
    dense4 = tf.keras.layers.Dense(16, activation='relu')(drop3)
    drop4 = tf.keras.layers.Dropout(dp)(dense4)

    predictions = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(drop4)
    
    model = tf.keras.Model(sequence_input, predictions)    
    

    #==================================================
    # Data random sampling
    #==================================================

    balancing_dict = Counter(np.argmax(y_train,axis = 1))
    for class_, obs_nb in balancing_dict.items():
        if obs_nb > 100:
            balancing_dict[class_] = 100
    
    
    rus = RandomUnderSampler(sampling_strategy = balancing_dict)
    ids = np.arange(len(X_train)).reshape((-1, 1))
    ids_rs, y_rs = rus.fit_sample(ids, y_train)
    X_rs = X_train[ids_rs.flatten()] 
    
    
    #==================================================
    # Specifying the optimizer
    #==================================================
  
    adam = tf.keras.optimizers.Adam(lr=1e-3)
        
    batch_size = {{choice([64, 128])}}
    STEP_SIZE_TRAIN = (len(X_rs) // batch_size) + 1 
    STEP_SIZE_VALID = (len(X_valid) // batch_size) + 1 

    alpha = {{choice([0.1, 0.25, 0.5, 0.8, 0.9])}}
    gamma = {{choice([0.5, 1, 1.5, 2, 2.5])}}

    model.compile(loss=[categorical_focal_loss(alpha= alpha, gamma = gamma)], metrics=['accuracy'], optimizer = adam)
    
    result = model.fit(X_rs, y_rs, validation_data=(X_valid, y_valid), \
                    steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                    epochs = 1, shuffle=True, verbose=2)

    #Get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_loss']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    os.chdir('C:/Users/rfuchs/Documents/GitHub/phyto_curves_reco')

    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=7,
                                          trials=Trials())
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = data()
    print("Evalutation of best performing model:")
    preds = best_model.predict(X_test)
    print(precision_score(y_test.argmax(1), preds.argmax(1), average = None, labels = list(range(y_test.shape[1]))))  
    print(confusion_matrix(y_test.argmax(1), preds.argmax(1), labels = list(range(y_test.shape[1]))))

    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    best_model.save('hyperopt_model_focal')
