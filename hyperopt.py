1# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 13:48:42 2020

@author: rfuchs
"""

import sys
import numpy as np
import pickle
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix, precision_score

import tensorflow as tf
from tensorflow_addons.optimizers import RectifiedAdam, Lookahead
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform, normal

from losses import CB_loss, categorical_focal_loss

#=================================================================
# Hyperoptimisation of the CNN
#=================================================================


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

    tn = pd.read_csv(data_dir + 'train_test_nomenclature.csv')

    X_train = train['X']
    X_valid = valid['X']
    X_test = test['X']


    y_train = train['y']
    y_valid = valid['y']
    y_test = test['y']

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
    filter_size = {{choice([3, 5, 7])}}
    x = tf.keras.layers.Conv1D(32, filter_size, activation='relu')(sequence_input)
    x = tf.keras.layers.Conv1D(16, filter_size, activation='relu')(x)
    x = tf.keras.layers.Conv1D(8, filter_size, activation='relu')(x)
    
    # Average the created features maps
    average = tf.keras.layers.GlobalAveragePooling1D()(x)
    dense2 = tf.keras.layers.Dense(32, activation='relu')(average) 
    drop2 = tf.keras.layers.Dropout(dp)(dense2)
    dense3 = tf.keras.layers.Dense(32, activation='relu')(drop2)
    drop3 = tf.keras.layers.Dropout(dp)(dense3)
    dense4 = tf.keras.layers.Dense(16, activation='relu')(drop3)
    drop4 = tf.keras.layers.Dropout(dp)(dense4)
    
    predictions = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(drop4)
    model = tf.keras.Model(sequence_input, predictions)


    #==================================================
    # Specifying the optimizer
    #==================================================
    model_dir = sys.argv[3]
    model_name = sys.argv[4]
    loss_name = sys.argv[5]

    weights_path = model_dir + 'weights_' + loss + '_' +  model_name + '.hdf5'
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    check = ModelCheckpoint(filepath = weights_path,\
                            verbose = 1, save_best_only=True)

    optim_ch = {{choice(['adam', 'ranger'])}}
    lr = {{uniform(1e-3, 1e-2)}} 

    if optim_ch == 'adam':
        optim = tf.keras.optimizers.Adam(lr = lr)
        sync_period = None
        slow_step_size = None
    else:
        sync_period = {{choice([1, 3])}}
        slow_step_size = {{normal(0.5, 0.1)}}
        rad = RectifiedAdam(lr = lr)
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
                    epochs = 50, class_weight = w, shuffle=True, verbose=2, callbacks = [check, es])
        
    elif loss_name == 'Class_balanced':
        beta = {{choice([0.9, 0.999, 0.9999])}} #remettre 0.99,0.99993
        gamma = {{uniform(0.5, 2.5)}}
    
        sample_per_class = np.sum(y_valid, axis = 0)
    
        model.compile(loss=[CB_loss(sample_per_class, beta = beta, gamma = gamma)],
                      metrics=['accuracy'], optimizer = optim)
        
    elif loss_name == 'Focal_loss':
        alpha = {{uniform(0, 1)}}
        gamma = {{uniform(0.5, 2.5)}}
        
        model.compile(loss=[categorical_focal_loss(alpha = alpha, gamma = gamma)],
                      metrics=['accuracy'], optimizer = optim)
    
        result = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), \
                        steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                        epochs = 30, shuffle=True, verbose=2, callbacks = [check, es])
    else:
        raise ValueError('Please enter a legal loss name')


    #Get the highest validation accuracy of the training epochs
    loss_acc = np.amin(result.history['val_loss'])
    print('Min loss of epoch:', loss_acc)
    model.load_weights(weights_path)
    return {'loss': loss_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    model_dir = sys.argv[3]
    model_name = sys.argv[4]

    best_run, best_model = optim.minimize(model=create_model,
                                          data = data,
                                          algo = tpe.suggest,
                                          max_evals = int(sys.argv[2]), 
                                          trials = Trials())
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = data()
    
    print("Evaluation of best performing model:")
    preds = best_model.predict(X_test)
    print('Micro accuracy: ', precision_score(y_test.argmax(1), preds.argmax(1), average = 'micro', labels = list(range(y_test.shape[1]))))
    print('Classes accuracy: ', precision_score(y_test.argmax(1), preds.argmax(1), average = None, labels = list(range(y_test.shape[1]))))
    print('Macro accuracy: ', precision_score(y_test.argmax(1), preds.argmax(1), average = 'macro', labels = list(range(y_test.shape[1]))))

    tn = pd.read_csv('/content/gdrive/My Drive/data/SWINGS/L1_retreated/67 percent_7 sets/train_test_nomenclature.csv')
    print('\n')
    pd.set_option("display.max_rows", None, "display.max_columns", None) 
    print(pd.DataFrame(confusion_matrix(y_test.argmax(1), preds.argmax(1),\
                            labels = tn['labels']), index = tn['cluster'], columns =  tn['cluster']))
   
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    with open('cnn_categ_best_params.pickle', 'wb') as handle:
        pickle.dump(model_dir + best_run, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    best_model.save(model_dir + model_name)
    print('best model saved in:', os.getcwd())
        
        