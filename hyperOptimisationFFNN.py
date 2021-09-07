# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:10:01 2020

@author: rfuchs
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd


from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform, normal

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow_addons.optimizers import RectifiedAdam, Lookahead

from losses import CB_loss, categorical_focal_loss

##############################################################################################
#################  Model 13 Hyper-parameters tuning on FUMSECK Data ##########################
##############################################################################################

def prec_rec_function(y_test, preds, cluster_classes, algo):
    ''' Compute the precision and recall for all classes'''
    prec = precision_score(y_test, preds, average=None)
    prec = dict(zip(cluster_classes, prec))
    prec['algorithm'] = algo
    
    recall= recall_score(y_test, preds, average=None)
    recall = dict(zip(cluster_classes, recall))
    recall['algorithm'] = algo
    
    return prec, recall



# Tuning include : Sampling strategy (RUS vs RUS + ENN)

def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """

    data_dir = sys.argv[1]

    train = np.load(data_dir + 'train.npz', allow_pickle = True)
    valid = np.load(data_dir + 'valid.npz', allow_pickle = True)
    test = np.load(data_dir + 'test.npz', allow_pickle = True)

    X_train = train['X']
    X_valid = valid['X']
    X_test = test['X']

    y_train = train['y']
    y_valid = valid['y']
    y_test = test['y']

    # Scale the data for numeric stability
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.fit_transform(X_valid)
    X_test = scaler.fit_transform(X_test)

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
    nb_features = X_train.shape[1]

    sequence_input = tf.keras.layers.Input(shape= nb_features, dtype='float32')


    dense2 = tf.keras.layers.Dense(32, activation='relu')(sequence_input) # Does using 2*32 layers make sense ?
    drop2 = tf.keras.layers.Dropout(dp)(dense2)
    dense3 = tf.keras.layers.Dense(16, activation='relu')(drop2)
    drop3 = tf.keras.layers.Dropout(dp)(dense3)
    dense4 = tf.keras.layers.Dense(8, activation='relu')(drop3)
    drop4 = tf.keras.layers.Dropout(dp)(dense4)

    predictions = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(drop4)

    model = tf.keras.Model(sequence_input, predictions)

    #==================================================
    # Specifying the optimizer
    #==================================================
    model_dir = sys.argv[3]
    model_name = sys.argv[4]
    loss_name = sys.argv[5]
    nb_epochs = int(sys.argv[6])
    
    weights_path = model_dir + 'weights_' + loss_name + '_' +  model_name + '.hdf5'
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    check = ModelCheckpoint(filepath = weights_path,\
                            verbose = 1, save_best_only=True)


    optim_ch = {{choice(['adam', 'ranger'])}}
    lr = {{uniform(1e-3, 1e-2)}} 

    if optim_ch == 'adam':
        optim = tf.keras.optimizers.Adam(learning_rate = lr)
        sync_period = None
        slow_step_size = None
    else:
        sync_period = {{choice([1, 3])}}
        slow_step_size = {{normal(0.5, 0.1)}}
        rad = RectifiedAdam(learning_rate = lr)
        optim = Lookahead(rad, sync_period = sync_period, slow_step_size = slow_step_size)

    # Batch size definition
    batch_size = {{choice([64 * 8, 64 * 4])}}
    STEP_SIZE_TRAIN = (len(X_train) // batch_size) + 1
    STEP_SIZE_VALID = (len(X_valid) // (64 * 8)) + 1
    
    #==============================================
    # Compile the model with the specified loss
    #==============================================
    
    if loss_name == 'categorical_crossentropy':
        # Defining the weights: Take the average over SSLAMM data
        weights = {{uniform(0, 1)}}
        w = 1 / (np.sum(y_valid, axis = 0)) ** weights
        w = w / w.sum()
        w = dict(zip(range(N_CLASSES),w))
        
        model.compile(loss='categorical_crossentropy',\
                      metrics=['accuracy'], optimizer=optim)

        result = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), \
                    steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                    epochs = nb_epochs, class_weight = w, shuffle=True, verbose=2,\
                    callbacks = [check, es])
        
    elif loss_name == 'Class_balanced':
        beta = {{choice([0.9, 0.999, 0.9999])}} #remettre 0.99,0.99993
        gamma = {{uniform(0.5, 2.5)}}
    
        sample_per_class = np.sum(y_valid, axis = 0)
    
        model.compile(loss=[CB_loss(sample_per_class, beta = beta, gamma = gamma)],
                      metrics=['accuracy'], optimizer = optim)
        
        result = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), \
                        steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                        epochs = nb_epochs, shuffle=True, verbose=2, callbacks = [check, es])
        
    elif loss_name == 'Focal_loss':
        alpha = {{uniform(0, 1)}}
        gamma = {{uniform(0.5, 2.5)}}
        
        model.compile(loss=[categorical_focal_loss(alpha = alpha, gamma = gamma)],
                      metrics=['accuracy'], optimizer = optim)
    
        result = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), \
                        steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                        epochs = nb_epochs, shuffle=True, verbose=2, callbacks = [check, es])
    else:
        raise ValueError('Please enter a legal loss name')


    #Get the highest validation accuracy of the training epochs
    loss_acc = np.amin(result.history['val_loss'])
    print('Min loss of epoch:', loss_acc)
    model.load_weights(weights_path)
    return {'loss': loss_acc, 'status': STATUS_OK, 'model': model}



if __name__ == '__main__':

    data_dir = sys.argv[1]
    model_dir = sys.argv[3]
    model_name = sys.argv[4]

    best_run, best_model = optim.minimize(model=create_model,
                                          data = data,
                                          algo = tpe.suggest,
                                          max_evals = int(sys.argv[2]), 
                                          trials = Trials())
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = data()

    tn = pd.read_csv(data_dir + 'train_test_nomenclature.csv')
    
    #======================================
    # Save the best model
    #======================================
    
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    
    with open(model_dir + model_name + '.pickle', 'wb') as handle:
        pickle.dump(best_run, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    best_model.save(model_dir + model_name)
    print('best model saved in:', os.getcwd())
  
    #======================================
    # Evaluate the best model on test data
    #======================================

    print("Evaluation of best performing model:")
    preds = best_model.predict(X_test)
    class_accuracy = precision_score(y_test.argmax(1), preds.argmax(1),\
                                     average = None, labels = list(range(y_test.shape[1])))

    print('Micro accuracy: ', precision_score(y_test.argmax(1), preds.argmax(1),\
                                    average = 'micro', labels = list(range(y_test.shape[1]))))
    print('Classes accuracy: ', dict(zip(tn['name'], class_accuracy)))
    print('Macro accuracy: ', precision_score(y_test.argmax(1), preds.argmax(1),\
                                    average = 'macro', labels = list(range(y_test.shape[1]))))

    print('\n')
    pd.set_option("display.max_rows", None, "display.max_columns", None) 
    print(pd.DataFrame(confusion_matrix(y_test.argmax(1), preds.argmax(1),\
                        labels = tn['id']), index = tn['name'], columns =  tn['name']))
   
        
