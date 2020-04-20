# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:39:14 2020

@author: rfuchs
"""

import re
import os
import pandas as pd
import numpy as np

# Fetch the files 
pred_folder =  "C:/Users/rfuchs/Documents/SSLAMM_P1/SSLAMM_L2"
pred_files = os.listdir(pred_folder)

pulse_regex = "Pulse" 
date_regex = "Pulse[0-9]{1,2}_(20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:u|h)[0-9]{2})"
flr_regex = 'Pulse([0-9]{1,2})'

files = [file for file in pred_files if re.search(pulse_regex, file)] # The files containing the data to predict
phyto_ts = pd.DataFrame(columns = ['picoeucaryote', 'synechococcus', 'nanoeucaryote', 'cryptophyte', \
       'noise', 'airbubble', 'microphytoplancton', 'prochlorococcus', 'date'])

for file in files:
    path = pred_folder + '/' + file
    
    cl_count = pd.read_csv(path, usecols = ['Pred FFT Label'])['Pred FFT Label'].value_counts()
    cl_count = pd.DataFrame(cl_count).transpose()
    flr_num = int(re.search(flr_regex, file).group(1))
    
    # Keep only "big" phyotplancton from FLR25 and "small" one from FLR6 
    if flr_num == 25:
        for clus_name in ['picoeucaryote', 'synechococcus', 'prochlorococcus']:
            if clus_name in cl_count.columns:
                cl_count[clus_name] = 0
        
    elif flr_num == 6:
        for clus_name in ['cryptophyte', 'nanoeucaryote', 'microphytoplancton']:
            if clus_name in cl_count.columns:
                cl_count[clus_name] = 0
    else:
        raise RuntimeError('Unkonwn flr number', flr_num)
    
    # The timestamp is rounded to the closest 20 minutes    
    date = re.search(date_regex, file).group(1)
    date = pd.to_datetime(date, format='%Y-%m-%d %Hh%M', errors='ignore')
    mins = date.minute
                
    if (mins >= 00) & (mins < 15): 
        date = date.replace(minute=00)

    elif (mins >= 15) & (mins <= 35): 
        date = date.replace(minute=20)
    
    elif (mins > 35) & (mins < 57):
        date = date.replace(minute=40)
        
    elif mins >= 57:
        if date.hour != 23:
            date = date.replace(hour= date.hour + 1, minute=00)
        else:
            try:
                date = date.replace(day = date.day + 1, hour = 00, minute=00)
            except:
                date = date.replace(month = date.month + 1, day = 1, hour = 00, minute=00)
                
           
    cl_count['date'] = date 
    phyto_ts = phyto_ts.append(cl_count)



##########################################################################
# On the fly prediction
##########################################################################

os.chdir('C:/Users/rfuchs/Documents/GitHub/phyto_curves_reco')

from pred_functions import pred_n_count
import tensorflow as tf
from tensorflow_addons.optimizers import Lookahead, RectifiedAdam
from time import time
from copy import deepcopy
from losses import categorical_focal_loss

# Model and nomenclature 
model = tf.keras.models.load_model('trained_models/hyperopt_model_focal2', compile = False)
model.compile(optimizer=Lookahead(RectifiedAdam(lr = 0.003589101299926042), 
                                  sync_period = 10, slow_step_size = 0.20736365316666247),
              loss = categorical_focal_loss(gamma = 2.199584705628343, alpha = 0.25))


tn = pd.read_csv('train_test_nomenclature.csv')
tn.columns = ['Particle_class', 'label']

phyto_ts = pd.DataFrame(columns = ['picoeucaryote', 'synechococcus', 'nanoeucaryote', 'cryptophyte', \
       'unassigned particle', 'airbubble', 'microphytoplancton', 'prochlorococcus', 'date'])

phyto_ts_proba = deepcopy(phyto_ts)
  

os.chdir('C:/Users/rfuchs/Documents/SSLAMM_P2')

  
# Define where to look the data at and where to store preds
'''
export_folder = "C:/Users/rfuchs/Documents/SSLAMM_P1/SSLAMM_unlab_compiled_L1"
files = os.listdir(export_folder)

pulse_regex = "_Pulse" 
files = [file for file in files if re.search(pulse_regex, file)] # The files containing the data to predict
'''

export_folder = "C:/Users/rfuchs/Documents/SSLAMM_P2"
files = os.listdir(export_folder)

pulse_regex = "_Pulse" 
files = [file for file in files if re.search(pulse_regex, file)] # The files containing the data to predict

count = 0
start = time()
for idx, file in enumerate(files):

    source_path = export_folder + '/' + file

    cl_count, cl_count_proba = pred_n_count(source_path, model, tn, exp_count = False)
    phyto_ts = phyto_ts.append(cl_count)
    phyto_ts_proba = phyto_ts_proba.append(cl_count_proba)

    count += 1
    
    if count % 10 == 0:
        print('Dumping file:', idx ,file)
        phyto_ts.to_csv('SSLAMM_count_17042020.csv')
        phyto_ts_proba.to_csv('SSLAMM_count_proba_17042020.csv', index = False)
    
        end = time() 
        print((end - start) / 60, 'hours ')


##########################################################################    
# Final serie with representative count for the first part of the serie
##########################################################################
os.chdir('C:/Users/rfuchs/Documents/preds/pred2/P1')
phyto_ts = pd.read_csv('SSLAMM_count_17042020.csv')
phyto_ts = phyto_ts.iloc[:,1:]

idx_pbs = pd.DataFrame(phyto_ts.groupby('date').size()) # Make the same thing for phyto_ts_proba 
idx_pbs = idx_pbs[idx_pbs[0] == 1].index

# Few date reformating
phyto_ts = phyto_ts.replace(pd.to_datetime('2019-09-18 14:20:00'),pd.to_datetime('2019-09-18 14:40:00'))
phyto_ts = phyto_ts.replace(pd.to_datetime('2019-11-09 07:00:00'),pd.to_datetime('2019-11-09 07:20:00'))
phyto_ts = phyto_ts.replace(pd.to_datetime('2019-11-14 09:40:00'),pd.to_datetime('2019-11-14 10:00:00'))
phyto_ts = phyto_ts.replace(pd.to_datetime('2019-11-21 14:00:00'),pd.to_datetime('2019-11-21 14:20:00'))
phyto_ts = phyto_ts.replace(pd.to_datetime('2019-12-10 14:40:00'),pd.to_datetime('2019-12-10 15:00:00'))

phyto_rpz_ts = phyto_ts.groupby('date').sum()
phyto_rpz_ts = phyto_rpz_ts.reset_index()

# For those which have not both a FLR6 and a FLR25 file, replace the missing values by NaN
for idx in idx_pbs:
    phyto_rpz_ts[phyto_rpz_ts['date'] == idx] = phyto_rpz_ts[phyto_rpz_ts['date'] == idx].replace(0, np.nan)

phyto_rpz_ts.to_csv('09_to_12_2019_proba.csv', index = False)

##########################################################################    
# Final serie with representative count for the second part of the serie
##########################################################################
   
idx_pbs = pd.DataFrame(phyto_ts.groupby('date').size()) # Make the same thing for phyto_ts_proba 
idx_pbs = idx_pbs[idx_pbs[0] == 1].index

# Few date reformating
phyto_ts = phyto_ts.replace(pd.to_datetime('2020-03-05 09:00:00'),pd.to_datetime('2020-03-05 09:20:00'))
phyto_ts = phyto_ts.replace(pd.to_datetime('2020-03-04 13:20:00'),pd.to_datetime('2020-03-04 13:40:00'))

phyto_rpz_ts = phyto_ts.groupby('date').sum()

phyto_rpz_ts = phyto_rpz_ts.reset_index()

# For those which have not both a FLR6 and a FLR25 file, replace the missing values by NaN
for idx in idx_pbs:
    phyto_rpz_ts[phyto_rpz_ts['date'] == idx] = phyto_rpz_ts[phyto_rpz_ts['date'] == idx].replace(0, np.nan)

phyto_rpz_ts.to_csv('C:/Users/rfuchs/Documents/02_to_03_2020.csv', index = False)