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
import fastparquet as fp

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
                except (pd.errors.EmptyDataError):
                    print('Empty dataset')
                    #continue

                                    
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
# Create "unbiased" datasets from all manual classifications (consensual voting)
#====================================================================== 

# To delete

# To recheck for the missing files. ex cryptophytes
unbiased_part = pd.DataFrame(columns = ['Particle ID', 'cluster', 'acq'])
cc_regex = '_([_ () 0-9A-Za-zµ]+)_Pulses.csv'

#acq = 'SSLAMM-FLR6 2019-10-05 09h59'
#expert = 'Marrec'

len_list = []

for acq in acquistion_names_lists:
    print(acq)
    cluster_ids = dict.fromkeys(homogeneous_cluster_names(cluster_classes), [])

    for expert_num, expert in enumerate(expert_names_list):
        print('Expert:', expert)

        #*******************************
        # Build a correspondance dict
        #******************************* 
        
        # To link the names of the files and the Original class
        expert_Pulse_files_names = os.listdir(pulse_dirs + '/' + expert)
        ccs_expert = [re.search(cc_regex, name).group(1) for name in expert_Pulse_files_names]
        ccs_expert = list(set(ccs_expert))
        ccs_expert_homogeneous = homogeneous_cluster_names(ccs_expert)
        
        names_corr = dict(zip(ccs_expert_homogeneous, ccs_expert))
        
        #*******************************
        # Open Pulse files
        #******************************* 
        pulses_files_acq = [name for name \
                            in expert_Pulse_files_names if re.search(acq, name)]
        pulses_files_acq = [name for name in pulses_files_acq \
                           if not(re.search('lock', name))]
        
        if len(pulses_files_acq) == 0:
            for cluster in cluster_ids.keys():
                cluster_ids[cluster] = []
                continue
            
        for cluster in cluster_classes:
            #print(cluster)
            file_name = acq + '_' + names_corr[cluster] + '_Pulses.csv'
                        
            if cluster == 'Default (all)':
                continue

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
            except FileNotFoundError:
                print(cluster, 'does not exists')
                cluster_ids[cluster] = []
                continue
            
            # Correct the cluster name
            cluster = homogeneous_cluster_names([cluster])[0]
                        
            # 0 is used as particle separation sign in Pulse shapes
            file = file[np.sum(file, axis = 1) != 0] 
             
            # Collect the particle IDs
            ids = np.unique(file['Particle ID'])
            
            # Keep only the particles that were clustered the same way
            if (len(cluster_ids[cluster]) == 0) & (expert_num == 0):
                cluster_ids[cluster] = deepcopy(ids)
            else:
                new_indices = set(cluster_ids[cluster]) - set(ids)
                cluster_ids[cluster] = list(set(cluster_ids[cluster]) - new_indices)
                
            #print('cluster nb particles= ', len(cluster_ids[cluster]))
                        
    # Compile all the remaining particles in a single DataFrame
    acq_pcles = pd.DataFrame(columns = ['Particle ID', 'cluster', 'acq'])
    for cc in cluster_ids.keys():
        df = pd.DataFrame(columns = ['Particle ID', 'cluster', 'acq'])
        df['Particle ID'] = cluster_ids[cc]
        df['cluster'] = cc
        df['acq'] = acq
        acq_pcles = acq_pcles.append(df)
        
    len_list.append(len(acq_pcles))
    unbiased_part = unbiased_part.append(acq_pcles)
    

    
unbiased_part.to_parquet('unbiased_particles.parq', compression = 'snappy',\
                         index = False)

#======================================================================
# Create "unbiased" datasets from all manual classifications (majority voting)
#====================================================================== 
from sklearn.preprocessing import OneHotEncoder

# In progress
# Check that for each file len(dummy) ==  len(default) 
# Some particles are not present for all participants
# Small diff with the preceding script

# To recheck for the missing files. ex cryptophytes
unbiased_part = pd.DataFrame(columns = ['Particle ID', 'cluster', 'acq'])
cc_regex = '_([_ () 0-9A-Za-zµ]+)_Pulses.csv'

#acq = 'SSLAMM-FLR6 2019-10-05 09h59'
#expert = 'Marrec'

nb_expert = len(expert_names_list)

len_list2 = []

for acq in acquistion_names_lists:
    print(acq)
    
    all_expert_pid_clusters = pd.DataFrame()
    for expert in expert_names_list:
        print('Expert:', expert)

        #*******************************
        # Build a correspondance dict
        #******************************* 
        
        # To link the names of the files and the Original class
        expert_Pulse_files_names = os.listdir(pulse_dirs + '/' + expert)
        ccs_expert = [re.search(cc_regex, name).group(1) for name in expert_Pulse_files_names]
        ccs_expert = list(set(ccs_expert))
        ccs_expert_homogeneous = homogeneous_cluster_names(ccs_expert)
        
        names_corr = dict(zip(ccs_expert_homogeneous, ccs_expert))
        
        #*******************************
        # Open Pulse files
        #******************************* 
        pulses_files_acq = [name for name \
                            in expert_Pulse_files_names if re.search(acq, name)]
        pulses_files_acq = [name for name in pulses_files_acq \
                           if not(re.search('lock', name))]
        
        if len(pulses_files_acq) == 0:
            continue
        
        #****************************************
        # Extract the particles of each cluster
        #****************************************
        
        pid_clusters = pd.DataFrame()
        for cluster in cluster_classes:
            #print(cluster)
            file_name = acq + '_' + names_corr[cluster] + '_Pulses.csv'
                        
            if cluster == 'Default (all)':
                continue

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
            except FileNotFoundError:
                print(cluster, 'does not exists')
                continue
            
            # Correct the cluster name
            cluster = homogeneous_cluster_names([cluster])[0]
                        
            # 0 is used as particle separation sign in Pulse shapes
            file = file[np.sum(file, axis = 1) != 0] 
            
            pid_cluster = deepcopy(pd.DataFrame(file['Particle ID']))
            pid_cluster['cluster'] = cluster   
            pid_cluster = pid_cluster.drop_duplicates()
            pid_clusters = pid_clusters.append(pid_cluster)
        
        #********************************************************************
        # Count how many times the particles have been counted in each class
        #********************************************************************
        dummy_pid_clusters = pd.get_dummies(pid_clusters.set_index('Particle ID'))
        all_expert_pid_clusters = all_expert_pid_clusters.add(dummy_pid_clusters, \
                                                              fill_value = 0)
            
        all_expert_pid_clusters = all_expert_pid_clusters.fillna(0)
       
    #********************************************************************
    # Keep the particles that have been similarly 
    # classified by a majority of experts
    #********************************************************************
        
    maj_voting_idx = (all_expert_pid_clusters > (nb_expert / 2)).sum(1) == 1
    all_expert_pid_clusters = all_expert_pid_clusters.loc[maj_voting_idx]
    all_expert_pid_clusters.columns = [re.sub('cluster_', '', \
                                              col) for col in all_expert_pid_clusters]
    df = all_expert_pid_clusters.idxmax(1).reset_index()
    df['acq'] = acq
    df.columns = ['Particle ID', 'cluster', 'acq']
    len_list2.append(len(df))
        
    unbiased_part = unbiased_part.append(df)
        
   
unbiased_part.to_parquet('unbiased_particles.parq', compression = 'snappy',\
                         index = False)
    
    

#======================================================================
# Create Pulses and Listmodes files from consensual particles
#====================================================================== 

pf = fp.ParquetFile('unbiased_particles.parq')
part_to_keep = pf.to_pandas()

#*********************************
# From Pulses
#*********************************

dest_repo = 'C:/Users/rfuchs/Documents/cyto_classif'

# Date and FLR regex
flr_regex = 'FLR([0-9]{1,2})'
date_regex = "(20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:h|u)[0-9]{2})"

for acq_idx, acq_part in part_to_keep.groupby('acq'):
    acq_name = np.unique(acq_part['acq'])[0]
    print(acq_name)
    
    flr_num = re.search(flr_regex, acq_name).group(1)
    date = re.search(date_regex, acq_name).group(1)
    
    
    expert_Pulse_files_names = os.listdir(pulse_dirs + '/Lotty')
            
    # Open Pulse files
    pulses_files_acq = [name for name \
                        in expert_Pulse_files_names if re.search(acq_name, name)]
    default_pulse = [name for name in pulses_files_acq \
                       if re.search('Default', name)][0]

    try:
        file = pd.read_csv(pulse_dirs + '/Lotty/' + \
                      default_pulse, sep = ';', dtype = np.float64)
    except ValueError: # If the data are in European format ("," stands for decimals and not thousands)
        try:
            file = pd.read_csv(pulse_dirs + '/Lotty/' + \
                      default_pulse, sep = ';', dtype = np.float64,\
                thousands='.', decimal=',')
        except pd.errors.EmptyDataError:
            print('Empty dataset')
            continue

    # 0 is used as particle separation sign in Pulse shapes
    file = file[np.sum(file, axis = 1) != 0] 
    
    # Label the unbiased data
    unbiased_parts = file.merge(acq_part[['Particle ID', 'cluster']])

    fp.write(dest_repo + '/XP_Pulses_L2' + '/Labelled_Pulse' + str(flr_num) + '_' +\
             date + '.parq', unbiased_parts, compression='SNAPPY')

#*********************************
# From Listmodes
#*********************************
os.chdir('C:/Users/rfuchs/Documents/These/Oceano/XP_biais')

list_dir = 'Listmodes'
pf = fp.ParquetFile('unbiased_particles.parq')
part_to_keep = pf.to_pandas()

dest_repo = 'C:/Users/rfuchs/Documents/cyto_classif'

# Date and FLR regex
flr_regex = 'FLR([0-9]{1,2})'
date_regex = "(20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:h|u)[0-9]{2})"

for acq_idx, acq_part in part_to_keep.groupby('acq'):
    acq_name = np.unique(acq_part['acq'])[0]
    print(acq_name)
    
    flr_num = re.search(flr_regex, acq_name).group(1)
    date = re.search(date_regex, acq_name).group(1)
    
    expert_List_files_names = os.listdir(list_dir)
            
    # Open Pulse files
    list_files_acq = [name for name \
                        in expert_List_files_names if re.search(acq_name, name)]
    default_list = [name for name in list_files_acq \
                       if re.search('Default', name)][0]

    try:
        file = pd.read_csv(list_dir + '/' +\
                      default_list, sep = ';', dtype = np.float64)
    except ValueError: # If the data are in European format ("," stands for decimals and not thousands)
        try:
            file = pd.read_csv(list_dir + '/' +\
                      default_list, sep = ';', dtype = np.float64,\
                thousands='.', decimal=',')
        except pd.errors.EmptyDataError:
            print('Empty dataset')
            continue
        
    # Label the unbiased data
    unbiased_parts = file.merge(acq_part[['Particle ID', 'cluster']], how = 'inner')
    
    print('Keep ', len(unbiased_parts), 'over', len(file), 'particles = ',\
          len(unbiased_parts) / len(file), '%')

    fp.write(dest_repo + '/XP_Listmodes' + '/Labelled_Pulse' + str(flr_num) + '_' +\
             date + '.parq', unbiased_parts, compression='SNAPPY')

