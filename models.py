# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:03:04 2019

@author: robin
"""

from keras.layers import Input, Dense, Conv1D, Concatenate, GlobalAveragePooling1D, \
            Dropout, MaxPooling1D, LSTM, Flatten
from keras.models import Model
from keras import optimizers

import matplotlib.pyplot as plt
import numpy as np

from keras import metrics

#=================================================================================================================#
# Predictions from curves values utils 
#=================================================================================================================#

def ffnn_model(X, y, dp = 0.2):
    ''' Create a Feed Forward Neural Net Model with dropout
    X (ndarray): The features
    y (ndarray): The labels 
    dp (float): The dropout rate of the model
    ---------------------------------------------------------
    returns (Keras Model): The compiled model 
    '''
    N_CLASSES = y.shape[1]
    max_len = nb_curves = X.shape[1]
    nb_curves = X.shape[2]
    
    sequence_input = Input(shape=(max_len, nb_curves), dtype='float32')
    
    average = GlobalAveragePooling1D()(sequence_input)
    dense1 = Dense(64, activation='relu')(average)
    drop1 = Dropout(dp)(dense1)
    dense2 = Dense(32, activation='relu')(drop1)
    drop2 = Dropout(dp)(dense2)
    dense3 = Dense(32, activation='relu')(drop2)
    drop3 = Dropout(dp)(dense3)
    dense4 = Dense(16, activation='relu')(drop3)
    drop4 = Dropout(dp)(dense4)

    predictions = Dense(N_CLASSES, activation='softmax')(drop4)
    
    model = Model(sequence_input, predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.01), metrics=['acc'])
    return model

def ffnn_model_w_len(X, y, seq_length, dp = 0.2):
    ''' Create a Feed Forward Neural Net Model with dropout
    X (ndarray): The features
    y (ndarray): The labels 
    seq_length (1d-array): The original length of the sequence, which is highly informative
    dp (float): The dropout rate of the model
    ---------------------------------------------------------
    returns (Keras Model): The compiled model 
    '''
    N_CLASSES = y.shape[1]
    max_len = nb_curves = X.shape[1]
    nb_curves = X.shape[2]
    
    sequence_input = Input(shape = (max_len, nb_curves), dtype='float32')
    length_input = Input(shape = (1,1), dtype = 'float32')
    
    # Extract features from the 5 curves
    average = GlobalAveragePooling1D()(sequence_input)
    dense1 = Dense(64, activation='relu')(average)
    drop1 = Dropout(dp)(dense1)
    dense2 = Dense(32, activation='relu')(drop1)
    drop2 = Dropout(dp)(dense2)
    dense3 = Dense(32, activation='relu')(drop2)
    drop3 = Dropout(dp)(dense3)
    dense4 = Dense(16, activation='relu')(drop3)
    drop4 = Dropout(dp)(dense4)

    flat_len = Flatten()(length_input)
    # Add the information about the sequence length
    combined = Concatenate(axis = -1)([drop4, flat_len])
    
    predictions = Dense(N_CLASSES, activation='softmax')(combined)
    
    model = Model([sequence_input, length_input], predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.01), metrics=['acc'])
    return model

def model13(X, y, dp = 0.2):
    ''' Create a Feed Forward Neural Net Model with dropout
    X (ndarray): The features
    y (ndarray): The labels 
    dp (float): The dropout rate of the model
    ---------------------------------------------------------
    returns (Keras Model): The compiled model 
    '''
    N_CLASSES = y.shape[1]
    max_len = X.shape[1]
    nb_curves = X.shape[2]
    
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
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.01), metrics=[metrics.categorical_accuracy])
    return model

def model13_light(X, y, dp = 0.2):
    ''' Create a Feed Forward Neural Net Model with dropout
    X (ndarray): The features
    y (ndarray): The labels 
    dp (float): The dropout rate of the model
    ---------------------------------------------------------
    returns (Keras Model): The compiled model 
    '''
    N_CLASSES = y.shape[1]
    max_len = nb_curves = X.shape[1]
    nb_curves = X.shape[2]
    
    sequence_input = Input(shape=(max_len, nb_curves), dtype='float32')
    
    # A 1D convolution with 128 output channels: Extract features from the curves
    x = Conv1D(32, 5, activation='relu')(sequence_input)
    x = Conv1D(16, 5, activation='relu')(x)

    # Average those features
    average = GlobalAveragePooling1D()(x)
    dense2 = Dense(32, activation='relu')(average)
    drop2 = Dropout(dp)(dense2)

    predictions = Dense(N_CLASSES, activation='softmax')(drop2)
    
    model = Model(sequence_input, predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.01), metrics=[metrics.categorical_accuracy])
    return model


def lstm_model(X, y):
    ''' Create a LSTM and Convolutional layers based model from O. Grisel Lecture-labs notebook
    X (ndarray): The features
    y (ndarray): The labels 
    ---------------------------------------------------------
    returns (Keras Model): The compiled model 
    '''
    N_CLASSES = y.shape[1]
    max_len = nb_curves = X.shape[1]
    nb_curves = X.shape[2]
    
    # input: a sequence of MAX_SEQUENCE_LENGTH integers
    sequence_input = Input(shape=(max_len, nb_curves), dtype='float32')
    
    # 1D convolution with 64 output channels
    x = Conv1D(64, 5)(sequence_input)
    
    # MaxPool divides the length of the sequence by 5: this is helpful
    # to train the LSTM layer on shorter sequences. The LSTM layer
    # can be very expensive to train on longer sequences.
    x = MaxPooling1D(5)(x)
    x = Conv1D(64, 5)(x)
    x = MaxPooling1D(5)(x)
    
    # LSTM layer with a hidden size of 64
    x = LSTM(64)(x)

    predictions = Dense(N_CLASSES, activation='softmax')(x)
    
    model = Model(sequence_input, predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.01), metrics=['acc'])
    return model

def conv_model(X, y):
    ''' Create a Convolutional layers based model
    X (ndarray): The features
    y (ndarray): The labels 
    ---------------------------------------------------------
    returns (Keras Model): The compiled model 
    '''
    N_CLASSES = y.shape[1]
    max_len = nb_curves = X.shape[1]
    nb_curves = X.shape[2]
    
    # input: a sequence of MAX_SEQUENCE_LENGTH integers
    sequence_input = Input(shape=(max_len, nb_curves), dtype='float32')
    
    # A 1D convolution with 128 output channels
    x = Conv1D(128, 5, activation='relu')(sequence_input)
    # MaxPool divides the length of the sequence by 5
    x = MaxPooling1D(5)(x)
    # A 1D convolution with 64 output channels
    x = Conv1D(64, 5, activation='relu')(x)
    # MaxPool divides the length of the sequence by 5
    x = MaxPooling1D(5)(x)
    x = Flatten()(x)
    
    predictions = Dense(N_CLASSES, activation='softmax')(x)
    
    model = Model(sequence_input, predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])
    return model


#===============================================================================
# General Keras plotting utility 
#===============================================================================

def plot_losses(history):
    ''' Plot the train and valid losses coming from the training of the model 
    history (Keras history): The history of the model while training
    ----------------------------------------------------------------
    returns (plt plot): The train and valid losses of the model through the epochs
    '''
    # list all data in history
    #print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()