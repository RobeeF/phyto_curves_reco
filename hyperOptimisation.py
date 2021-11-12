# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 13:48:42 2020

@author: rfuchs
"""

import os
import sys
import numpy as np
import pickle
import pandas as pd
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

    X_train = train['X']
    X_valid = valid['X']

    y_train = train['y']
    y_valid = valid['y']
    
    return X_train, y_train, X_valid, y_valid


def create_model(X_train, y_train, X_valid, y_valid):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """

    N_CLASSES = y_train.shape[1]
    max_len = X_train.shape[1]
    nb_curves = X_train.shape[2]

    sequence_input = tf.keras.layers.Input(shape=(max_len, nb_curves), dtype='float32')

    # A 1D convolution with 128 output channels: Extract features from the curves
    kernel_size = 3 # Earlier on: 3, 5, 7
    x = tf.keras.layers.Conv1D(filters = 32, kernel_size = kernel_size, activation='relu')(sequence_input)
    x = tf.keras.layers.Conv1D(filters = 32, kernel_size = kernel_size, activation='relu')(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=2, strides=None, padding="valid", data_format="channels_last")(x)
    x = tf.keras.layers.Conv1D(filters = 64, kernel_size = kernel_size, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters = 64, kernel_size = kernel_size, activation='relu')(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=2, strides=None, padding="valid", data_format="channels_last")(x)
    x = tf.keras.layers.Conv1D(filters = 128, kernel_size = kernel_size, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters = 128, kernel_size = kernel_size, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters = 128, kernel_size = kernel_size, activation='relu')(x)
    
    # Average the created features maps
    average = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')(x)
    dense2 = tf.keras.layers.Dense(216, activation='relu')(average) 

    predictions = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(dense2)
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

    optim_ch = 'ranger'
    lr = {{uniform(1e-3, 1e-2)}} 

    sync_period = {{choice([1, 3])}}
    slow_step_size = {{normal(0.5, 0.1)}}
    rad = RectifiedAdam(learning_rate = lr)
    optim = Lookahead(rad, sync_period = sync_period, slow_step_size = slow_step_size)

    # Batch size definition
    batch_size = {{choice([64 * 2, 64 * 4])}} 
    STEP_SIZE_TRAIN = (len(X_train) // batch_size) + 1
    STEP_SIZE_VALID = (len(X_valid) // (64 * 8)) + 1
    
    #==============================================
    # Compile the model with the specified loss
    #==============================================
    
    if loss_name == 'categorical_crossentropy':
        
        w = np.full(N_CLASSES, 1 / N_CLASSES)
        w = dict(zip(range(N_CLASSES),w))

        
        model.compile(loss='categorical_crossentropy',\
                      metrics=['accuracy'], optimizer=optim)

        result = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), \
                    steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                    epochs = nb_epochs, class_weight = w, shuffle=True, verbose=2,\
                    callbacks = [check, es])
     
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
    
    tn = pd.read_csv(data_dir + 'train_test_nomenclature.csv') 
    test = np.load(data_dir + 'test.npz', allow_pickle = True)
    X_test = test['X']
    y_test = test['y']

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