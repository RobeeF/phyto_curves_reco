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
  

# Extracted from X_test
thrs = [0.8158158158158159, 0.7297297297297297, 0.5085085085085085, 0.3963963963963964, 0.8378378378378378, \
        0.7417417417417418, 0.42542542542542544]

  
# Define where to look the data at and where to store preds
os.chdir('C:/Users/rfuchs/Documents/SSLAMM_P1')

export_folder = "C:/Users/rfuchs/Documents/SSLAMM_P1/SSLAMM_unlab_compiled_L1"
files = os.listdir(export_folder)

pulse_regex = "_Pulse" 
files = [file for file in files if re.search(pulse_regex, file)] # The files containing the data to predict

'''
os.chdir('C:/Users/rfuchs/Documents/SSLAMM_P2')

export_folder = "C:/Users/rfuchs/Documents/SSLAMM_P2"
files = os.listdir(export_folder)

pulse_regex = "_Pulse" 
files = [file for file in files if re.search(pulse_regex, file)] # The files containing the data to predict
'''

count = 0
start = time()
for idx, file in enumerate(files):

    source_path = export_folder + '/' + file

    cl_count = pred_n_count(source_path, model, tn, thrs, exp_count = False)
    phyto_ts = phyto_ts.append(cl_count)
    #phyto_ts_proba = phyto_ts_proba.append(cl_count_proba)

    count += 1
    
    if count % 10 == 0:
        print('Dumping file:', idx ,file)
        phyto_ts.to_csv('SSLAMM_count_17042020.csv')
        #phyto_ts_proba.to_csv('SSLAMM_count_proba_17042020.csv', index = False)
    
        end = time() 
        print((end - start) / 60, 'minutes')


##########################################################################    
# Final serie with representative count for the first part of the serie
##########################################################################

os.chdir('C:/Users/rfuchs/Documents/preds/pred3/P1')
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

phyto_rpz_ts.to_csv('09_to_12_2019.csv', index = False)

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









# New version: 







import re
import os
import pandas as pd
import numpy as np

# Fetch the files 
pred_folder =  r"C:\Users\rfuchs\Documents\cyto_classif\SSLAMM_P1\Preds_P1"
pred_files = os.listdir(pred_folder)

pulse_regex = "Pulse" 
date_regex = "Pulse[0-9]{1,2}_(20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:u|h)[0-9]{2})"
flr_regex = 'Pulse([0-9]{1,2})'

files = [file for file in pred_files if re.search(pulse_regex, file)] # The files containing the data to predict





sum_files_repo = "C:/Users/rfuchs/Documents/cyto_classif/SSLAMM_P1/summary"

phyto_ts = pd.DataFrame(columns = ['airbubble', 'cryptophyte', 'nanoeucaryote',\
                   'inf1microm_unidentified_particle', 'microphytoplancton',\
                'picoeucaryote', 'prochlorococcus', \
                'sup1microm_unidentified_particle', 'synechococcus', 'date', 'FLR'])


SWS_noise_thr = 70
FWS_crypto_thr = 1E4
FWS_micros_thr = 2 * 10 ** 5

unwanted_files = []


for file in files:
    if file in unwanted_files:
      print(file)
      continue
      
    path = pred_folder + '/' + file
    cl_count = pd.read_csv(path, usecols = ['Pred FFT Label', 'Total FWS', 'Total SWS'])

    # For cryptos, micros and phrochlo post processing
    real_cryptos = np.logical_and(cl_count['Total FWS'] >= FWS_crypto_thr, cl_count['Pred FFT Label'] == 'cryptophyte').sum()
    real_microphytos = np.logical_and(cl_count['Total FWS'] >= FWS_micros_thr, cl_count['Pred FFT Label'] == 'microphytoplancton').sum()
    false_microphytos = np.logical_and(cl_count['Total FWS'] < FWS_micros_thr, cl_count['Pred FFT Label'] == 'microphytoplancton').sum()
    false_noise = ((cl_count['Total FWS'] <= 100) & (cl_count['Total SWS'] >= SWS_noise_thr) & (cl_count['Pred FFT Label'] == 'inf1microm_unidentified_particle')).sum()
    #
    
    cl_count = cl_count['Pred FFT Label'].value_counts()
 
    cl_count = pd.DataFrame(cl_count).transpose()
    flr_num = int(re.search(flr_regex, file).group(1))
    
    # Keep only "big" phyotplancton from FLR25 and "small" one from FLR6 
    if flr_num == 25:
        for clus_name in ['synechococcus', 'prochlorococcus']:
            if clus_name in cl_count.columns:
                cl_count[clus_name] = 0
        
        # Post processing rules
        cl_count['cryptophyte'] = real_cryptos
        cl_count['microphytoplancton'] = real_microphytos
        
        try:
            cl_count['nanoeucaryote'] += false_microphytos
        except KeyError:
          cl_count['nanoeucaryote'] = false_microphytos

      
    elif flr_num == 6:
        for clus_name in ['picoeucaryote', 'cryptophyte', 'nanoeucaryote', 'microphytoplancton']:
            if clus_name in cl_count.columns:
                cl_count[clus_name] = 0
        
        
        if file == 'Pulse6_2019-11-08 09h58.csv':
            print('Before postprocessing:')
            print(cl_count['prochlorococcus'])
        
        # Post processing rules
        try:
          cl_count['prochlorococcus'] += false_noise
        except KeyError:
          cl_count['prochlorococcus'] = false_noise

    else:
        raise RuntimeError('Unknown flr number', flr_num)

    if file == 'Pulse6_2019-11-08 09h58.csv':
        print('----------------')
        print('After postprocessing')
        print(cl_count['prochlorococcus'])

    try:
        cl_count['inf1microm_unidentified_particle'] -= false_noise
    except KeyError:
        cl_count['inf1microm_unidentified_particle'] = 0


    # Extract the date
    date = re.search(date_regex, file).group(1)

    # Compute the volume
    sep = '-' if flr_num == 6 else '_'
    sum_file = 'SSLAMM' + sep + 'FLR' + str(flr_num) + ' ' + date + '_Info.txt'

    try:
      with open(sum_files_repo + '/' + sum_file) as f:
        lines = f.readlines()

        try:
            vol = re.search('Volume .+: ([0-9]+,[0-9]+)', str(lines)).group(1)
            vol = float(re.sub(',', '.', vol))
        except AttributeError:
            vol = np.nan
            raise RuntimeError('Pas trouvé')
    except FileNotFoundError:
      print(file)
      continue

    cl_count = cl_count.div(int(vol))

    # The timestamp is rounded to the closest 20 minutes    
    date = pd.to_datetime(date, format='%Y-%m-%d %Hh%M', errors='ignore')
    mins = date.minute
        

    if (mins >= 00) & (mins <= 30): 
        date = date.replace(minute=00)



    elif (mins >= 31): # On arrondit à l'heure d'après
        if date.hour != 23:
            date = date.replace(hour= date.hour + 1, minute=00)
        else:
            try:
                date = date.replace(day = date.day + 1, hour = 00, minute=00)
            except:
                date = date.replace(month = date.month + 1, day = 1, hour = 00, minute=00)
    else:
      raise RuntimeError(date,'non handled')
           
    cl_count['date'] = date 
    cl_count['FLR'] = flr_num 

    phyto_ts = phyto_ts.append(cl_count)
    
    
idx_pbs = pd.DataFrame(phyto_ts.groupby(['date', 'FLR']).size()) # Make the same thing for phyto_ts_proba 
idx_pbs = idx_pbs[idx_pbs[0] > 1].index
idx_pbs = [id_[0] for id_ in  idx_pbs]

phyto_ts_ok = phyto_ts[~phyto_ts['date'].isin(idx_pbs)]

phyto_ts_resolved_pbs =  phyto_ts[phyto_ts['date'].isin(idx_pbs)].set_index(['date', 'FLR']).groupby(['date', 'FLR']).max().reset_index() # Take the more likely entry
phyto_ts = phyto_ts_ok.reset_index(drop = True).append(phyto_ts_resolved_pbs)


idx_pbs = pd.DataFrame(phyto_ts.groupby('date').size()) # Make the same thing for phyto_ts_proba 
idx_pbs = idx_pbs[idx_pbs[0] == 1].index


phyto_rpz_ts = phyto_ts.groupby('date').sum()
phyto_rpz_ts = phyto_rpz_ts.reset_index()

# For those which have not both a FLR6 and a FLR25 file, replace the missing values by NaN
for idx in idx_pbs:
    phyto_rpz_ts[phyto_rpz_ts['date'] == idx] = phyto_rpz_ts[phyto_rpz_ts['date'] == idx].replace(0, np.nan)
    
    
phyto_rpz_ts['noise'] = phyto_rpz_ts['inf1microm_unidentified_particle'] + phyto_rpz_ts['sup1microm_unidentified_particle']
del(phyto_rpz_ts['inf1microm_unidentified_particle'])
del(phyto_rpz_ts['sup1microm_unidentified_particle'])
#del(phyto_rpz_ts['FLR'])

phyto_rpz_ts[(phyto_rpz_ts == 0)] = np.nan

# Delete an outlier
phyto_rpz_ts['prochlorococcus'] = np.where(phyto_rpz_ts['prochlorococcus'] <= 60, phyto_rpz_ts['prochlorococcus'], np.nan)


phyto_rpz_ts.to_csv(r'C:\Users\rfuchs\Documents\preds\pred4\P1\09_to_12_2019_concentration.csv', index = False)

#==================================================================
# Create ts from P1 to P5
#==================================================================

from copy import deepcopy

ts_P1 = pd.read_csv('C:/Users/rfuchs/Documents/preds/pred4/P1/09_to_12_2019_concentration.csv')
ts_P2 = pd.read_csv('C:/Users/rfuchs/Documents/preds/pred4/P2/02_to_06_2020_concentration.csv')
ts_P3 = pd.read_csv('C:/Users/rfuchs/Documents/preds/pred4/P3/06_to_07_2020_concentration.csv')
ts_P4 = pd.read_csv('C:/Users/rfuchs/Documents/preds/pred4/P4/07_to_10_2020_concentration.csv')
ts_P5 = pd.read_csv('C:/Users/rfuchs/Documents/preds/pred4/P5/10_to_12_2020_concentration.csv')

ts = ts_P1.append(ts_P2).append(ts_P3).append(ts_P4).append(ts_P5)
#ts = ts[['date'] + interesting_classes]

# Delete the most obvious outliers
#max_proch  = ts.prochlorococcus.max()
#ts['prochlorococcus'] = ts.prochlorococcus.replace({max_proch: np.nan})

ts.to_csv('C:/Users/rfuchs/Documents/preds/pred4/P1_P5_concentration.csv', index = True)
