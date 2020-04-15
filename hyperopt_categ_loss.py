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

    tn = pd.read_csv('train_test_nomenclature.csv')
    tn.columns = ['Particle_class', 'label']
    
    X_train = np.load('SSLAMM_L3/X_train.npy')
    y_train = np.load('SSLAMM_L3/y_train.npy')

    X_valid = np.load('SSLAMM_L3/X_valid.npy')
    y_valid = np.load('SSLAMM_L3/y_valid.npy') 

    X_test = np.load('SSLAMM_L3/X_test.npy')
    y_test = np.load('SSLAMM_L3/y_test.npy')    
            
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
    
    sequence_input = Input(shape=(max_len, nb_curves), dtype='float32')
    
    # A 1D convolution with 128 output channels: Extract features from the curves
    x = Conv1D(64, 5, activation='relu')(sequence_input)
    x = Conv1D(32, 5, activation='relu')(x)
    x = Conv1D(16, 5, activation='relu')(x)

    # Average those features
    average = GlobalAveragePooling1D()(x)
    dense2 = Dense(32, activation='relu')(average) # Does using 2*32 layers make sense ?
    drop2 = Dropout(dp)(dense2)
    dense3 = Dense(32, activation='relu')(drop2)
    drop3 = Dropout(dp)(dense3)
    dense4 = Dense(16, activation='relu')(drop3)
    drop4 = Dropout(dp)(dense4)

    predictions = Dense(N_CLASSES, activation='softmax')(drop4)
    
    
    model = Model(sequence_input, predictions)
    
    #==================================================
    # Data random sampling
    #==================================================

    balancing_dict = Counter(np.argmax(y_train,axis = 1))
    for class_, obs_nb in balancing_dict.items():
        if obs_nb > 500:
            balancing_dict[class_] = 500
    
    
    rus = RandomUnderSampler(sampling_strategy = balancing_dict)
    ids = np.arange(len(X_train)).reshape((-1, 1))
    ids_rs, y_rs = rus.fit_sample(ids, y_train)
    X_rs = X_train[ids_rs.flatten()] 
    
    
    #==================================================
    # Specifying the optimizer
    #==================================================
  
    adam = optimizers.Adam(lr={{choice([10**-3, 10**-2, 10**-1])}})
    ada = optimizers.Adadelta(lr={{choice([10**-3, 10**-2, 10**-1])}})
    sgd = optimizers.SGD(lr={{choice([10**-3, 10**-2, 10**-1])}})
    
    choiceval = {{choice(['adam', 'sgd', 'ada'])}}
    if choiceval == 'adam':
        optim = adam
    elif choiceval == 'ada':
        optim = ada
    else:
        optim = sgd
        
    # Defining the weights: Take the average over SSLAMM data
    weights = {{choice(['regular', 'sqrt'])}}
    
    if weights == 'regular':
        w = 1 / np.sum(y_valid, axis = 0)
        w = w / w.sum()
        
    else:
        w = 1 / np.sqrt(np.sum(y_valid, axis = 0))
        w = w / w.sum() 

    batch_size = {{choice([64, 128])}}
    STEP_SIZE_TRAIN = (len(X_rs) // batch_size) + 1 
    STEP_SIZE_VALID = (len(X_valid) // batch_size) + 1 


    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optim)
    
    result = model.fit(X_rs, y_rs, validation_data=(X_valid, y_valid), \
                    steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                    epochs = 10, class_weight = w, shuffle=True, verbose=2)

    #Get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_loss']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}




if __name__ == '__main__':
    os.chdir('C:/Users/rfuchs/Documents/GitHub/planktonPipeline/extract_Pulse_values')

    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = data()
    print("Evalutation of best performing model:")
    preds = best_model.predict(X_test)
    print(precision_score(y_test.argmax(1), preds.argmax(1), average = None, labels = list(range(y_test.shape[1]))))   
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    best_model.save('hyperopt_model_categ')
