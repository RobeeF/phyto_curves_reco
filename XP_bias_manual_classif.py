# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 11:33:52 2020

@author: rfuchs
"""

import os 
import re
import numpy as np
import pandas as pd
from copy import deepcopy

os.chdir('C:/Users/rfuchs/Documents/GitHub/phyto_curves_reco')
from dataset_preprocessing import homogeneous_cluster_names


os.chdir('C:/Users/rfuchs/Documents/These/Oceano/XP_biais')


count_dirs = 'Counts/'
pulse_dirs = 'Pulse_shapes/'

expert_names_list = os.listdir(count_dirs)
acquistion_names_lists = ['SSLAMM-FLR6 2019-10-05 09h59',
 'SSLAMM-FLR6 2019-12-08 15h59',
 'SSLAMM-FLR6 2020-02-19 05h59',
 'SSLAMM-FLR6 2020-04-30 19h59',
 'SSLAMM-FLR6 2020-06-26 11h59',
 'SSLAMM-FLR6 2020-06-28 01h59',
 'SSLAMM_FLR25 2019-10-05 10h07',
 'SSLAMM_FLR25 2019-12-08 16h07',
 'SSLAMM_FLR25 2020-02-19 06h07',
 'SSLAMM_FLR25 2020-04-30 20h07',
 'SSLAMM_FLR25 2020-06-26 12h07',
 'SSLAMM_FLR25 2020-06-28 02h07']


expert = expert_names_list[0]
acq = acquistion_names_lists[0]
cc_regex = '_([_ () 0-9A-Za-zµ]+)_Pulses.csv'

cluster_classes = ['airbubble', 'cryptophyte', 'nanoeucaryote',\
                   'inf1microm_unidentified_particle', 'microphytoplancton',\
                'picoeucaryote', 'prochlorococcus', \
                'sup1microm_unidentified_particle', 'synechococcus']
    

#======================================================================
# Compute average counts and std (biases) over manual classifications
#====================================================================== 
# What about spe count ? noise in FLR6 or FLR25 ?
# CAREFUL: Missing files for Marrec !

count_df = pd.DataFrame(columns = ['cluster', 'Count', 'expert', 'acq'])

for expert in expert_names_list:
    print('Expert counts actually collected:', expert)

    for acq in acquistion_names_lists:
        print(acq)

        try:
            counts = pd.read_csv(count_dirs + '/' + expert + \
                                     '/' + acq + '.csv', sep = ';')
        except FileNotFoundError:
            print('Could not import ', acq)
            continue
            
        # Check that the name of the file is the good one
        assert re.search(acq, counts['Filename'][0])
        
        # Make the cluster names homogeneous
        counts.columns = ['Filename', 'cluster', 'Count', 'Concentration']
        counts = homogeneous_cluster_names(counts)
        counts = counts.groupby(['cluster']).sum().reset_index()
        
        # Add the missing classes 
        for cc in cluster_classes:
            if not(cc in list(counts['cluster'])):
                counts = counts.append({'cluster': cc, 'Count': 0},\
                                       ignore_index=True)
        counts['expert'] = expert
        counts['acq'] = acq
        
        count_df = count_df.append(counts)

count_df['Count'] = count_df['Count'].astype(int)  
count_df.groupby(['cluster', 'acq']).agg({'Count': np.mean}) 
count_df.groupby(['cluster', 'acq']).agg({'Count': np.std}) 


#======================================================================
# Check counts computed by Cytoclus == counts from Pulse shapes
#====================================================================== 

for expert in expert_names_list:
    print('Expert checked for file compliance:', expert)
    expert_Pulse_files_names = os.listdir(pulse_dirs + '/' + expert )

    for acq in acquistion_names_lists:
        print(acq)
        
        #*******************************
        # Read the counts files
        #*******************************
        try:
            counts = pd.read_csv(count_dirs + '/' + expert + \
                                 '/' + acq + '.csv', sep = ';')
        except FileNotFoundError:
            print('Could not import ', acq)
            continue
        
        # Check that the name of the file is the good one
        assert re.search(acq, counts['Filename'][0])
        
        # Make the cluster names homogeneous
        counts.columns = ['Filename', 'cluster', 'Count', 'Concentration']
        counts = homogeneous_cluster_names(counts)
        counts = counts.groupby(['cluster']).sum().reset_index()
        
        # Add the missing classes 
        for cc in cluster_classes:
            if not(cc in list(counts['cluster'])):
                counts = counts.append({'cluster': cc, 'Count': 0},\
                                       ignore_index=True)
        
        #*******************************
        # Recount from the Pulse shapes
        #******************************* 
        pulse_counts = pd.DataFrame(columns = ['cluster', 'Count']) 

        pulses_files_acq = [name for name \
                            in expert_Pulse_files_names if re.search(acq, name)]
        pulses_files_acq = [name for name in pulses_files_acq \
                           if not(re.search('lock', name))]
        
        # Recount the number of particle in each cluster
        for file_name in pulses_files_acq:
            cluster = re.search(cc_regex, file_name).group(1)
            
            try:
                file = pd.read_csv(pulse_dirs + '/' + expert + \
                         '/' + file_name, sep = ';', dtype = np.float64)
            except ValueError: # If the data are in European format ("," stands for decimals and not thousands)
                try:
                    file = pd.read_csv(pulse_dirs + '/' + expert + \
                         '/' + file_name, sep = ';', dtype = np.float64,\
                        thousands='.', decimal=',')
                except pd.errors.EmptyDataError:
                    print('Empty dataset')
                    continue
            
                        
            # 0 is used as particle separation sign in Pulse shapes
            file = file[np.sum(file, axis = 1) != 0] 
             
            ids = np.unique(file['Particle ID'])
            cc_count= len(ids)
            
            pulse_counts = pulse_counts.append({'cluster': cluster,\
                            'Count': cc_count}, ignore_index=True)

        pulse_counts = homogeneous_cluster_names(pulse_counts)
        pulse_counts = pulse_counts.groupby(['cluster']).sum().reset_index()
        
        # Add the missing classes 
        for cc in cluster_classes:
            if not(cc in list(pulse_counts['cluster'])):
                pulse_counts = pulse_counts.append({'cluster': cc, 'Count': 0},\
                                       ignore_index=True)
        
        counts_comp = counts.merge(pulse_counts, on = 'cluster')
        counts_dif = np.abs(counts_comp['Count_x'] - counts_comp['Count_y']).sum()
        
        if counts_dif:
            print("error")
            print(counts_comp)
            raise RuntimeError('Count difference')
            print('--------------------------------------')
        

#======================================================================
# Create "unbiased" datasets from all manual classifications 
#====================================================================== 

acq_dict = pd.DataFrame(columns = ['Particle ID', 'cluster', 'acq'])

for acq in acquistion_names_lists:
    print(acq)
    cluster_ids = dict.fromkeys(cluster_classes)

    for expert in expert_names_list:
        print('Expert:', expert)
        #expert_Pulse_files_names = os.listdir(pulse_dirs + '/' + expert )
            
        #*******************************
        # Open Pulse files
        #******************************* 

        pulses_files_acq = [name for name \
                            in expert_Pulse_files_names if re.search(acq, name)]
        pulses_files_acq = [name for name in pulses_files_acq \
                           if not(re.search('lock', name))]
        
        # Recount the number of particle in each cluster
        for file_name in pulses_files_acq:
            cluster = re.search(cc_regex, file_name).group(1)
            print(cluster)

            try:
                file = pd.read_csv(pulse_dirs + '/' + expert + \
                         '/' + file_name, sep = ';', dtype = np.float64)
            except ValueError: # If the data are in European format ("," stands for decimals and not thousands)
                try:
                    file = pd.read_csv(pulse_dirs + '/' + expert + \
                         '/' + file_name, sep = ';', dtype = np.float64,\
                        thousands='.', decimal=',')
                except pd.errors.EmptyDataError:
                    print('Empty dataset')
                    continue
            
            # Correct the cluster name
            cluster = homogeneous_cluster_names([cluster])[0]
                        
            # 0 is used as particle separation sign in Pulse shapes
            file = file[np.sum(file, axis = 1) != 0] 
             
            # Collect the particle IDs
            ids = np.unique(file['Particle ID'])
            
            # Keep only the ones that were already found
            if len(cluster_ids[cluster]):
                cluster_ids[cluster] = deepcopy(ids)
            else:
                # To check
                new_indices = set(cluster_ids[cluster]) - set(ids)
                cluster_ids[cluster] = list(set(cluster_ids[cluster]) - new_indices)
                
            
    # Work to do here            
    acq_dict.append(cluster_ids)
            
            
            
