# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:17:25 2019

@author: Utilisateur
"""

import os

os.chdir('C:/Users/rfuchs/Documents/GitHub/phyto_curves_reco')


import re
import pandas as pd

from pred_functions import predict
from keras.models import load_model



# Model and nomenclature loading
model = load_model('trained_models/LottyNet_FUMSECK')

tn = pd.read_csv('train_test_nomenclature.csv')
tn.columns = ['Particle_class', 'label']

# Define where to look the data at and where to store preds
export_folder = "C:/Users/rfuchs/Documents/SSLAMM_P2/SSLAMM_L1"
export_files = os.listdir(export_folder)

pulse_regex = "_Pulse" 
files_to_pred = [file for file in export_files if re.search(pulse_regex, file)] # The files containing the data to predict

# Create a log file in the destination folder: list of the already predicted files
preds_store_folder = "C:/Users/rfuchs/Documents/SSLAMM_P2/SSLAMM_L2"  # Where to store the predictions
log_path = preds_store_folder + "/pred_logs.txt" # Register where write the already predicted files

if not(os.path.isfile(log_path)):
    open(preds_store_folder + '/pred_logs.txt', 'w+').close()

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
        predict(path, preds_store_folder,  model, tn, is_ground_truth = False)
        
        # Write in the logs that this file is already predicted
        with open(log_path, "a") as log_file:
            log_file.write(file + '\n')
            
    else:
        print(file, 'already predicted')
        