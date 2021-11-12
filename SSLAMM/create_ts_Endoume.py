# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:39:14 2020

@author: rfuchs
"""

import re
import os
import pandas as pd
import numpy as np


os.chdir('C:/Users/rfuchs/Documents/GitHub/phyto_curves_reco')
from pred_functions import combine_files_into_acquisitions, post_processing


# Fetch the files 
pred_folder = 'C:/Users/rfuchs/Documents/GitHub/phytoUpwelling/datasets/preds/SSLAMM_preds'
sum_files_repo = 'C:/Users/rfuchs/Documents/GitHub/phytoUpwelling/datasets/preds/SSLAMM_summary'

pulse_regex = "Pulse" 
date_regex = "Pulse[0-9]{1,2}_(20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:u|h)[0-9]{2})"
flr_regex = 'Pulse([0-9]{1,2})' #flr5 ou flr25

pred_files = os.listdir(pred_folder)
files = [file for file in pred_files if re.search(pulse_regex, file)] # The files containing the data to predict
  
phyto_ts = pd.DataFrame(columns = ['airbubble', 'cryptophyte', 'nanoeucaryote',\
                   'inf1microm_unidentified_particle', 'microphytoplancton',\
                'picoeucaryote', 'prochlorococcus', \
                'sup1microm_unidentified_particle', 'synechococcus', 'date', 'FLR'])

for idx, file in enumerate(files):
    if (idx % 100) == 0:
        print(idx, ' files have already been processed')
        
    flr_num = int(re.search(flr_regex, file).group(1))
    path = pred_folder + '/' + file
    cl_count = pd.read_csv(path, usecols = ['Pred FFT Label', 'Total FWS', 'Total SWS']) #lit le fichier et le classe en trois colonnes
    cl_count = post_processing(cl_count, flr_num)    #dans pred_functions, change la classe de certaines particules 
    cl_count = cl_count['Pred FFT Label'].value_counts() #compte le nombre de particules dans chaque classe
    cl_count = pd.DataFrame(cl_count).transpose() #met sous forme de tableau+échange lignes et colonnes

    # Keep only "big" phyotplancton from FLR25 and "small" one from FLR6 
    if flr_num == 25:
      cl_count[['synechococcus', 'prochlorococcus']] = 0
    else:
      cl_count[['picoeucaryote', 'cryptophyte', 'nanoeucaryote', 'microphytoplancton']] = 0
    
    # Compute the volume
    sep = '-' if flr_num == 6 else '_'
    date = re.search(date_regex, file).group(1)  
    sum_file = 'SSLAMM' + sep + 'FLR' + str(flr_num) + ' ' + date + '_Info.txt'
      
    try:
      with open(sum_files_repo + '/' + sum_file) as f:
        lines = f.readlines()
        vol = re.search('Volume .+: ([0-9]+,[0-9]+)', str(lines)).group(1)
        vol = float(re.sub(',', '.', vol))

    except FileNotFoundError:
      print(file, ' volume was not found')
      continue

    # If the volume analyzed is too small then the quality of the data is 
    # not guaranteed
    if (flr_num == 25) & (vol <= 1500):
        continue
    if (flr_num == 6) & (vol <= 350):
        continue
    
     
    # Compute the concentration (also called abundance)
    cl_count = cl_count.div(int(vol)) #divise toutes les colonnes par le volume total -> donne la concentration

    # The timestamp is rounded to the closest hour
    date = pd.to_datetime(date, format='%Y-%m-%d %Hh%M', errors='ignore')
    date = date.round('2H')
           
    cl_count['date'] = date #ce qu'on met dans la colonne date
    cl_count['FLR'] = flr_num 

    # New format:
    phyto_ts = phyto_ts.append(cl_count)

phyto_ts = combine_files_into_acquisitions(phyto_ts) #les prédictions FLR5 et 20 sont combinées

del(phyto_ts['inf1microm_unidentified_particle'])
del(phyto_ts['sup1microm_unidentified_particle'])

phyto_ts = phyto_ts.reset_index()

phyto_ts.to_csv('C:/Users/rfuchs/Documents/GitHub/phytoUpwelling/datasets/INSITU/raw/P1_P5_abundance.csv',\
                index = False)
    
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
