# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:21:55 2020

@author: rfuchs
"""


from __future__ import print_function
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe

from hyperas import optim
from hyperas.distributions import choice, uniform, normal

import pickle

from collections import Counter
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
#from keras.layers import Input, Conv1D,  GlobalAveragePooling1D, Dense, Dropout
#from keras.models import Model
from keras import metrics
import tensorflow as tf

#from keras import optimizers

from sklearn.metrics import confusion_matrix, precision_score
from losses import categorical_focal_loss

from tensorflow_addons.optimizers import RectifiedAdam, Lookahead
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import StandardScaler


###############################################################################
#################  FFNN tuning on FUMSECK Data ##########################
###############################################################################

# Tuning include : Sampling strategy (RUS vs RUS + ENN)

def data():
    """
    Data providing function:

    This function is separated from create_model() so that hyperopt
    won't reload data for each evaluation run.
    """

    train = np.load('train.npz', allow_pickle = True)
    valid = np.load('valid.npz', allow_pickle = True)
    test = np.load('test.npz', allow_pickle = True)

    X_train = train['X']
    X_valid = valid['X']
    X_test = test['X']

    y_train = train['y']
    y_valid = valid['y']
    y_test = test['y']

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.fit_transform(X_valid)
    X_test = scaler.fit_transform(X_test)

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
    nb_features = X_train.shape[1]

    sequence_input = tf.keras.layers.Input(shape= nb_features, dtype='float32')

    dense2 = tf.keras.layers.Dense(64, activation='relu')(sequence_input) # Does using 2*32 layers make sense ?
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

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    check = ModelCheckpoint(filepath='w_focal_hyperopt.hdf5',\
                            verbose = 1, save_best_only=True)

    optim_ch = {{choice(['adam', 'ranger'])}}
    lr = {{uniform(1e-3, 1e-2)}}

    if optim_ch == 'adam':
        optim = tf.keras.optimizers.Adam(lr = lr)
    else:
        sync_period = {{choice([2, 6, 10])}}
        slow_step_size = {{normal(0.5, 0.1)}}
        rad = RectifiedAdam(lr = lr)
        optim = Lookahead(rad, sync_period = sync_period, slow_step_size = slow_step_size)

    batch_size = {{choice([64 * 4, 64 * 8])}}
    STEP_SIZE_TRAIN = (len(X_train) // batch_size) + 1
    STEP_SIZE_VALID = 1


    alpha = {{uniform(0, 1)}}
    gamma = {{uniform(1, 2.5)}}

    model.compile(loss=[categorical_focal_loss(alpha= alpha, gamma = gamma)],
                  metrics=['accuracy'], optimizer = optim)

    result = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), \
                    steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                    epochs = 60, shuffle=True, verbose=2, callbacks = [check, es])

    #Get the highest validation accuracy of the training epochs
    loss_acc = np.amin(result.history['val_loss'])
    print('Min loss of epoch:', loss_acc)
    model.load_weights('w_focal_hyperopt.hdf5')
    return {'loss': loss_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':

    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=30,
                                          trials=Trials())

    X_train, y_train, X_valid, y_valid, X_test, y_test = data()
    print("Evalutation of best performing model:")
    preds = best_model.predict(X_test)
    print(precision_score(y_test.argmax(1), preds.argmax(1), average = 'micro', labels = list(range(y_test.shape[1]))))
    print(precision_score(y_test.argmax(1), preds.argmax(1), average = None, labels = list(range(y_test.shape[1]))))
    print(precision_score(y_test.argmax(1), preds.argmax(1), average = 'macro', labels = list(range(y_test.shape[1]))))

    print(confusion_matrix(y_test.argmax(1), preds.argmax(1), labels = list(range(y_test.shape[1]))))

    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    with open('ffnn_focal_best_params.pickle', 'wb') as handle:
        pickle.dump(best_run, handle, protocol=pickle.HIGHEST_PROTOCOL)
    best_model.save('ffnn_hyperopt_model_focal')
