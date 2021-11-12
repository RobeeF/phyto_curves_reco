# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:17:25 2019

@author: Utilisateur
"""

import os

os.chdir('C:/Users/rfuchs/Documents/GitHub/phyto_curves_reco')


import re
import pandas as pd

from pred_functions import predict, format_data
from tensorflow.keras.models import load_model, model_from_json
from keras.models import load_model, model_from_json


import tensorflow as tf
tf.__version__


dp = 0.2

N_CLASSES = 9
max_len = 120
nb_curves = 5

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

import json 

with open('trained_models/king_cnn.txt') as json_file:
    config = json.load(json_file)
    
model_from_json(config)

# Model and nomenclature loading
model = load_model('trained_models/king_cnn')

tn = pd.read_csv('train_test_nomenclature.csv')
tn.columns = ['Particle_class', 'label']

# Define where to look the data at and where to store preds
export_folder = "C:/Users/rfuchs/Documents/SSLAMM_P1/SSLAMM_unlab_compiled_L1"
export_files = os.listdir(export_folder)

pulse_regex = "_Pulse" 
files_to_pred = [file for file in export_files if re.search(pulse_regex, file)] # The files containing the data to predict

precomputed_data_dir = 'C:/Users/rfuchs/Documents/precomp/P1'
log_path = precomputed_data_dir + "/pred_logs.txt" # Register where write the already predicted files

# Create a log file in the destination folder: list of the already predicted files
preds_store_folder = "C:/Users/rfuchs/Documents/preds_files/P1"  # Where to store the predictions
#log_path = preds_store_folder + "/pred_logs.txt" # Register where write the already predicted files

if not(os.path.isfile(log_path)):
    open(log_path, 'w+').close()


for file in files_to_pred:
    print('Currently predicting ' + file)
    path = export_folder + '/' + file
    is_already_pred = False
    
    # Check if file has already been predicted
    with open(log_path, "r") as log_file:
        if file in log_file.read(): 
            is_already_pred = True
            
    if not(is_already_pred): # If not, perform the prediction
        # Predict the values
        #format_data(path, precomputed_data_dir, scale = False, \
                #is_ground_truth = False, hard_store = True)
        predict(path, preds_store_folder,  model, tn, \
            is_ground_truth = False, precomputed_data_dir = precomputed_data_dir)

        # Write in the logs that this file is already predicted
        with open(log_path, "a") as log_file:
            log_file.write(file + '\n')
            
    else:
        print(file, 'already predicted')
        