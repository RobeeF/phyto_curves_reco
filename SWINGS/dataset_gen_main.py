# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 09:43:43 2021

@author: rfuchs
"""

import os
import numpy as np
import pandas as pd
import fastparquet as fp
import scipy.integrate as it
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler 

code_dir = 'C:/Users/rfuchs/Documents/GitHub/'
XP_dir = r'C:/Users/rfuchs/Documents/These/Oceano/XP_bias_SWINGS/'

os.chdir(code_dir)

from phyto_curves_reco.dataset_preprocessing import centroid_sampling,\
    gen_dataset, quick_preprocessing
from phyto_curves_reco.viz_functions import plot_2Dcyto
from SWINGS.utilities import select_particles  

# Regular expressions:
flr_regex = 'FLR([0-9]{1,2})'
datetime_regex = "(20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}h[0-9]{2})"

pulse_dir = XP_dir + 'Pulse_shapes/'
parq_dir = XP_dir + 'L1/'
consensus_dir = 'consensual_PIDs/'
L2_dir = 'L2/'

cluster_classes = ['MICRO', 'ORGNANO', 'ORGPICOPRO', 'REDNANO',\
                   'REDPICOEUK', 'REDPICOPRO',  'inf1microm','sup1microm']
    
# Encode the cluster classes
le = LabelEncoder()
le.fit(cluster_classes)
cluster_classes

# Take 3 files out for the test set
files = os.listdir(parq_dir + 'Pulse')
#test_files_names = np.random.choice(files, 3)
# !!! TO REMOVE LATER ON and add a seed
seed = None
    
#'MAP-IO-SWINGS-FLR5 2021-01-27 00h11.parq'
test_files_names = ['MAP-IO-SWINGS-FLR5 2021-01-09 12h43.parq',\
              'MAP-IO-SWINGS-FLR5 2021-02-06 20h11.parq',\
              'MAP-IO-SWINGS-FLR20 2021-01-28 13h59.parq']    

tn = pd.read_csv(XP_dir + 'train_test_nomenclature.csv')
    
#===============================================
# Test set generation
#===============================================

#::::::::::::::::::::::::::::
# Pulse shape
#::::::::::::::::::::::::::::

X_test, seq_len_list_test, y_test, pid_list_test, test_files, le_test = gen_dataset(source = parq_dir + 'Pulse/', \
                        le = le, cluster_classes = cluster_classes, files = test_files_names, nb_obs_to_extract_per_group = 1E10, \
                          to_balance = False, seed = None)

np.savez_compressed(XP_dir + L2_dir + 'Pulse/test.npz', X = X_test, y = y_test)


#::::::::::::::::::::::::::::
# Listmode
#::::::::::::::::::::::::::::

X_test = pd.DataFrame()
y_test = []

for file in test_files_names:
    print(file)
    pfile = fp.ParquetFile(parq_dir + 'Listmodes/' + file)
    df = pfile.to_pandas()

    # Split X and y
    X_test_file, y_test_file = df.iloc[:,:-1], df['cluster']

    # Stack it with the previous file data 
    X_test = pd.concat([X_test, X_test_file])
    y_test = y_test + y_test_file.to_list()

# Convert y into a 0/1 matrix 
y_test = le.transform(y_test)
y_test = to_categorical(y_test, num_classes = len(cluster_classes))

del(X_test['Particle ID'])
X_test = X_test.reset_index(drop = True)

np.savez_compressed(XP_dir + L2_dir + 'Listmodes/test.npz', X = X_test, y = y_test)

   
#valid = np.load(XP_dir + L2_dir + 'Listmodes/train.npz')
#len(valid['X'])
    
#===============================================
# Compute the integrated curves DataFrame
#===============================================

files = list(set(files) - set(test_files_names))

all_integrated_curves = pd.DataFrame()

for file in files:
    print(file)
    pfile = fp.ParquetFile(parq_dir + 'Pulse/' + file)
    df = pfile.to_pandas()
    
    df = df.reset_index().groupby('Particle ID').agg(
    {
          'FWS': it.trapz,
          'SWS': it.trapz,
          'Fl Yellow': it.trapz,
          'FL Red': it.trapz,
          'Curvature': it.trapz, 
          'cluster': lambda x: list(x)[0] # Fetch the name of the cluster for each particle   
    })
    
    # Keep track of the origin of each particle
    df = df.reset_index('Particle ID')
    df['Filename'] = file
    
    all_integrated_curves = all_integrated_curves.append(df)


all_integrated_curves.to_csv(XP_dir + 'total.csv', index = False)
del(all_integrated_curves)


#=====================================================
# Valid set: Stratified undersampling
#=====================================================
total = pd.read_csv(XP_dir + 'total.csv')

# Clean interpolated curves:
X = total[['Particle ID', 'Filename', 'FWS','SWS', 'Fl Yellow', 'FL Red', 'Curvature']]
X = X.set_index(['Filename', 'Particle ID'])

# Labels:
y = total.set_index(['Filename', 'Particle ID'])['cluster']

del(total)


#**********************************
# Select the particle IDS
#**********************************

p_nonvalid = 0.8703
X_nonvalid_total, X_valid_total, y_nonvalid_total, y_valid_total = train_test_split(X, y,\
                                    train_size = p_nonvalid, random_state = 0)

# Free some memory
del(X)
del(y)

Counter(y_valid_total)


#**********************************
# Fetch the corresponding curves
#**********************************

#::::::::::::::::::::::::::::
# Pulse shape
#::::::::::::::::::::::::::::

max_len = 120
n_curves = 5

y_valid = []
X_valid = np.empty((0, max_len, n_curves))

for file in files:
    print(file)
    # Store the selected PIDs
    X_valid_total_file = X_valid_total.reset_index()
    pids = X_valid_total_file.loc[X_valid_total_file['Filename'] == file]['Particle ID']
    
    # Do not take into account the useless files
    if len(pids) == 0:
      print('No selected particles from:', file)
      continue
    
    # Load the parq and keep the selected particles
    f = fp.ParquetFile(parq_dir + 'Pulse/' + file)
    f = f.to_pandas().reset_index()
    file_curves = f[f['Particle ID'].isin(pids)].set_index('Particle ID')
        
    # Format the curves
    X_valid_file, y_valid_file = quick_preprocessing(file_curves,\
                                    ['FWS', 'SWS', 'Fl Yellow', 'FL Red', 'Curvature'])
    
    # Stack it with the previous file data 
    X_valid = np.concatenate([X_valid, X_valid_file])
    y_valid = y_valid + y_valid_file
  

# Convert y into a 0/1 matrix 
y_valid = le.transform(y_valid)
y_valid = to_categorical(y_valid, num_classes = len(cluster_classes))


np.savez_compressed(XP_dir + L2_dir + 'Pulse/valid.npz', X = X_valid, y = y_valid)


#::::::::::::::::::::::::::::
# Listmode
#::::::::::::::::::::::::::::
    
y_valid = []
X_valid = pd.DataFrame()

for file in files:
    print(file)
    # Store the selected PIDs
    X_valid_total_file = X_valid_total.reset_index()
    pids = X_valid_total_file.loc[X_valid_total_file['Filename'] == file]['Particle ID']
    
    # Do not take into account the useless files
    if len(pids) == 0:
      print('No selected particles from:', file)
      continue
    
    # Load the parq and keep the selected particles
    f = fp.ParquetFile(parq_dir + 'Listmodes/'  + file)
    f = f.to_pandas()
    file_curves = f[f['Particle ID'].isin(pids)]  
    
    # Split X and y
    X_valid_file, y_valid_file = file_curves.iloc[:,:-1], file_curves['cluster']
    
    # Stack it with the previous file data 
    X_valid = pd.concat([X_valid, X_valid_file])
    y_valid = y_valid + y_valid_file.to_list()

# Convert y into a 0/1 matrix 
y_valid = le.transform(y_valid)
y_valid = to_categorical(y_valid, num_classes = len(cluster_classes))

del(X_valid['Particle ID'])
X_valid = X_valid.reset_index(drop = True)

np.savez_compressed(XP_dir + L2_dir + 'Listmodes/valid.npz', X = X_valid, y = y_valid)


#===============================================
# Train set: Undersampling based on centroid distance
#===============================================

#**********************************
# Select the particle IDS
#**********************************
train_counts = Counter(y_nonvalid_total)
sampling_strategy = {cluster: np.min([count, 7000]) for cluster, count in train_counts.items()}

# Random Undersampling 
rus = RandomUnderSampler(sampling_strategy = sampling_strategy, random_state = 0)
indices = pd.DataFrame(X_nonvalid_total.index.to_list(), columns = ['Filename', 'Particle ID'])
X_train_total_idx, y_train_total = rus.fit_resample(indices, y_nonvalid_total)
indices = pd.MultiIndex.from_frame(X_train_total_idx, names = X_nonvalid_total.index.names)
X_train_total = X_nonvalid_total.loc[indices]
y_train_total.index = X_train_total.index


#****************************************
# Reinforce low represented zones
#****************************************
# Fetch the unselected curves
left_particles_indices = pd.DataFrame(list(set(X_nonvalid_total.index) - set(indices)),\
                                      columns = ['Filename', 'Particle ID'])
left_particles_indices = pd.MultiIndex.from_frame(left_particles_indices, names = X_nonvalid_total.index.names)
left_particles = X_nonvalid_total.loc[left_particles_indices]
left_labels = y_nonvalid_total.loc[left_particles_indices]

remaining_orgnanos = np.sum(left_labels == 'ORGNANO')

# Define the sampling strategies
sampling_addings = [['inf1microm', [10 ** 1, 10 ** 3], [10 ** 1, 8 * 10 ** 1], 1000, 'FWS', 'FL Red'],\
                    ['inf1microm', [10 ** 2, 10 ** 3], [10 ** 1, 8 * 10 ** 1], 300, 'FWS', 'FL Red'],\
                    ['REDPICOPRO', [10 ** 1, 10 ** 3], [10 ** 1, 8 * 10 ** 1], 1000, 'FWS', 'FL Red'],\
                    ['inf1microm', [10 ** 1, 5 * 10 ** 2], [10 ** 2, 6 * 10 ** 3], 500, 'FWS', 'FL Red'],\
                    ['REDPICOEUK', [10 ** 3, 10 ** 5], [2 * 10 ** 2, 10 ** 4], 1000, 'FWS', 'FL Red'],\
                    ['sup1microm', [3 * 10 ** 3, 2 * 10 ** 4], [5 * 10 ** 2, 10 ** 6], 500, 'FL Red', 'Fl Yellow'],\
                    ['ORGNANO', [10 ** 1, 10 ** 6], [10 ** 1, 10 ** 6], remaining_orgnanos, 'FWS', 'FL Red'],\
                    ['REDNANO', [10 ** 1, 10 ** 6], [10 ** 1, 10 ** 6], 1000, 'FWS', 'FL Red']]

    
for adding in sampling_addings:
    sampled_particles, sampled_labels = select_particles(left_particles, left_labels, adding[0], adding[1],\
                 adding[2], adding[3], random_state = 0, q1 = adding[4], q2 = adding[5])
    X_train_total = X_train_total.append(sampled_particles)
    y_train_total = y_train_total.append(sampled_labels)

# Centroid sampling for ORGNANOS
sampling_strategy = {cluster: 0 for cluster, count in train_counts.items()}
sampling_strategy['ORGNANO'] = 5000

X_train_orgnanos, y_train_orgnanos = centroid_sampling(X_train_total, y_train_total, sampling_strategy,\
                                                 columns = ['Fl Yellow', 'FL Red'],\
                                                 random_state = 0)

# Keep the closest ORGNANO to the centroid
X_train_total = X_train_total[y_train_total != 'ORGNANO']
X_train_total = X_train_total.append(X_train_orgnanos)
y_train_total = y_train_total[y_train_total != 'ORGNANO']
y_train_total = y_train_total.append(y_train_orgnanos['cluster'])


Counter(y_train_total)

plot_2Dcyto(X_train_total, y_train_total, 'FWS', 'FL Red')      
plot_2Dcyto(X_valid_total, y_valid_total, 'FWS', 'FL Red')


# Free the memory
del(X_nonvalid_total)
del(y_nonvalid_total)


#**********************************
# Fetch the corresponding curves
#**********************************

#::::::::::::::::::::::::::::
# Pulse shape
#::::::::::::::::::::::::::::
    
max_len = 120
n_curves = 5

y_train = []
X_train = np.empty((0, max_len, n_curves))

for file in files:
    print(file)
    # Store the selected PIDs
    X_train_total_file = X_train_total.reset_index()
    pids = X_train_total_file.loc[X_train_total_file['Filename'] == file]['Particle ID']
    
    # Do not take into account the useless files
    if len(pids) == 0:
      print('No selected particles from:', file)
      continue
    
    # Load the parq and keep the selected particles
    f = fp.ParquetFile(parq_dir + 'Pulse/'  + file)
    f = f.to_pandas().reset_index()
    file_curves = f[f['Particle ID'].isin(pids)].set_index('Particle ID')
    
    # Format the curves
    X_train_file, y_train_file = quick_preprocessing(file_curves, ['FWS', 'SWS', 'Fl Yellow', 'FL Red', 'Curvature'])
    
    assert len(X_train_file) == len(set(file_curves.index))
    
    # Stack it with the previous file data 
    X_train = np.concatenate([X_train, X_train_file])
    y_train = y_train + y_train_file

# Convert y into a 0/1 matrix 
y_train = le.transform(y_train)
y_train = to_categorical(y_train, num_classes = len(cluster_classes))

np.savez_compressed(XP_dir + L2_dir + 'Pulse/train.npz', X = X_train, y = y_train)

#::::::::::::::::::::::::::::
# Listmode
#::::::::::::::::::::::::::::
    
X_train = pd.DataFrame()
y_train = []

for file in files:
    print(file)
    # Store the selected PIDs
    X_train_total_file = X_train_total.reset_index()
    pids = X_train_total_file.loc[X_train_total_file['Filename'] == file]['Particle ID']
    
    # Do not take into account the useless files
    if len(pids) == 0:
      print('No selected particles from:', file)
      continue
    
    # Load the parq and keep the selected particles
    f = fp.ParquetFile(parq_dir + 'Listmodes/'  + file)
    f = f.to_pandas()
    file_curves = f[f['Particle ID'].isin(pids)]  
    
    # Split X and y
    X_train_file, y_train_file = file_curves.iloc[:,:-1], file_curves['cluster']
    
    # Stack it with the previous file data 
    X_train = pd.concat([X_train, X_train_file])
    y_train = y_train + y_train_file.to_list()

# Convert y into a 0/1 matrix 
y_train = le.transform(y_train)
y_train = to_categorical(y_train, num_classes = len(cluster_classes))

del(X_train['Particle ID'])
X_train = X_train.reset_index(drop = True)

np.savez_compressed(XP_dir + L2_dir + 'Listmodes/train.npz', X = X_train, y = y_train)
    

#===============================================
# Print the composition of all sets
#===============================================

# Import the labels
y_train = np.load(XP_dir + L2_dir + 'Pulse/train.npz')['y'].argmax(1)
y_valid =  np.load(XP_dir + L2_dir + 'Pulse/valid.npz')['y'].argmax(1)
y_test = np.load(XP_dir + L2_dir + 'Pulse/test.npz')['y'].argmax(1)

# Create a DataFrame with all the PFGs for the 3 sets
train_compo = pd.DataFrame(pd.Series(Counter(y_train)), columns = ['train'])
valid_compo = pd.DataFrame(pd.Series(Counter(y_valid)), columns = ['valid'])
test_compo = pd.DataFrame(pd.Series(Counter(y_test)), columns = ['test'])
set_composition = train_compo.join(valid_compo).join(test_compo).sort_index()

# Label the groups rather than using their index
set_composition.index = tn['name']
print(set_composition)
set_composition.index = set_composition.index.str.capitalize()
set_composition.to_csv(XP_dir + L2_dir + 'composition.csv')
set_composition.to_latex(XP_dir + L2_dir + 'composition.tex')

set_composition.sum(0)
