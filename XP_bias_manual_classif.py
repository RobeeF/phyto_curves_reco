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
import matplotlib.pyplot as plt
from scipy.integrate import trapz


os.chdir('C:/Users/rfuchs/Documents/GitHub/phyto_curves_reco')
from dataset_preprocessing import homogeneous_cluster_names
from viz_functions import plot_2Dcyto


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


enriched_files = ['SSLAMM-FLR6 2019-12-08 15h59',
 'SSLAMM-FLR6 2020-02-19 05h59',
 'SSLAMM-FLR6 2020-06-26 11h59',
 'SSLAMM_FLR25 2019-12-08 16h07',
 'SSLAMM_FLR25 2020-02-19 06h07',
 'SSLAMM_FLR25 2020-06-26 12h07']



# Regular expressions:
flr_regex = 'FLR([0-9]{1,2})'
date_regex = "(20[0-9]{2}-[0-9]{2}-[0-9]{2})"

expert = expert_names_list[0]
acq = acquistion_names_lists[0]
cc_regex = '[0-9]_([_\-() 0-9A-Za-zµ]+)_Pulses.csv'

cluster_classes = ['airbubble', 'cryptophyte', 'nanoeucaryote',\
                   'inf1microm_unidentified_particle', 'microphytoplancton',\
                'picoeucaryote', 'prochlorococcus', \
                'sup1microm_unidentified_particle', 'synechococcus']

classes_of_interest = ['airbubble', 'cryptophyte', 'nanoeucaryote',\
                'microphytoplancton', 'picoeucaryote', 'prochlorococcus', \
                'synechococcus', 'noise']

# Noise is useless and would have to account for unmarked particles
small_cells = ['picoeucaryote', 'synechococcus', 'prochlorococcus', 'airbubble']
large_cells = ['cryptophyte', 'nanoeucaryote', 'microphytoplancton', 'airbubble']

def creach_post_processing(acq, data):
    ''' For some of Creach 's acquisitions low_fluo correspond to < 1 micrometre
    noise and to >1 micrometre noise for others '''

    lowfluo1_pat = '[_A-Za-z0-9?\-()]*low[ _]{0,1}fluo1[_A-Za-z()0-9?\-]*'
    lowfluo2_pat = '[_A-Za-z0-9?\-()]*low[ _]{0,1}fluo2[_A-Za-z()0-9?\-]*'
    
    acqs_lowfluo1_is_inf1um = ['SSLAMM_FLR25 2020-06-28 02h07', 'SSLAMM_FLR25 2020-06-26 12h07']
    acqs_low_fluo2_is_inf1um = ['SSLAMM-FLR6 2019-10-05 09h59', 'SSLAMM-FLR6 2020-06-28 01h59',\
                                'SSLAMM_FLR25 2019-10-05 10h07', 'SSLAMM_FLR25 2020-04-30 20h07',\
                                'SSLAMM_FLR25 2020-06-26 12h07']
        
    if type(data) == pd.core.frame.DataFrame: 
        # Correct lowfluo1
        if acq in acqs_lowfluo1_is_inf1um: 
            data['cluster'] = data['cluster'].str.replace(lowfluo1_pat,\
                            'inf1microm_unidentified_particle', regex = True,\
                                case = False) 
        
        # Correct lowfluo2    
        
        if acq in acqs_low_fluo2_is_inf1um: # Add regex
            data['cluster'] = data['cluster'].str.replace(lowfluo2_pat,\
                            'inf1microm_unidentified_particle', regex = True,\
                                case = False)
        else:
            data['cluster'] = data['cluster'].str.replace(lowfluo2_pat,\
                            'sup1microm_unidentified_particle', regex = True,\
                                case = False)        

    else:
        # Correct lowfluo1
        if acq in acqs_lowfluo1_is_inf1um: # Case sensitive ?
            data = [re.sub(lowfluo1_pat,'inf1microm_unidentified_particle', string)\
                    for string in data]

        # Correct lowfluo2    
        
        if acq in acqs_low_fluo2_is_inf1um: # Add regex
            data = [re.sub(lowfluo2_pat,'inf1microm_unidentified_particle', string)\
                    for string in data]

        else:
            data = [re.sub(lowfluo2_pat,'sup1microm_unidentified_particle', string)\
                    for string in data]
    return data


#======================================================================
# Compute average counts and std (biases) over manual classifications
#====================================================================== 
# Airbubbles on FLR6 or FLR25 ?
# Latimier mettre un Default
# Check Lotty 1microm parsing error
# Check mean computation does not take missing values at the numerator

# ATTENTION FAIRE OPERATION PAR DATE DANS CETTE PARTIE, A RECHECKER !!!
# AJOUTER LE NOISE

temp_dir = 'C:/Users/rfuchs/Documents/These/Oceano/XP_biais/temp_count2/'

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
        
        # Need small post-processing for Creach's name
        if expert == 'Creach':
            counts = creach_post_processing(acq, counts)

        counts = homogeneous_cluster_names(counts)        
        counts = counts.groupby(['cluster']).sum().reset_index()
        
        # Add the missing classes 
        for cc in cluster_classes:
            if not(cc in list(counts['cluster'])):
                counts = counts.append({'cluster': cc, 'Count': 0},\
                                       ignore_index=True)
                    
        # Group all the noise into one class
        counts['cluster'] = counts.cluster.str.replace('sup1microm_unidentified_particle','noise')
        counts['cluster'] = counts.cluster.str.replace('inf1microm_unidentified_particle','noise')
        counts['cluster'] = counts.cluster.str.replace('unidentified_particle','noise')

        # Report expert and acquisition names
        counts['expert'] = expert
        counts['acq'] = acq
        
        count_df = count_df.append(counts)

print(np.unique(count_df["cluster"]))
count_df['Count'] = count_df['Count'].astype(int)  

# Keep only the sets belonging to the interesting classes
#count_df = count_df[count_df['cluster'].isin(classes_of_interest)] 

#******************
# Merge the FLR6 and FLR25 files of the same day
#******************
count_df['flr'] = count_df['acq'].str.extract(flr_regex).astype(int)
count_df['date'] = count_df['acq'].str.extract(date_regex)
count_df['enriched'] = count_df['acq'].isin(enriched_files)

# Special extract
spe_extract_6 = (count_df['flr'] == 6) & (count_df['cluster'].isin(small_cells))
spe_extract_25 = (count_df['flr'] == 25) & (count_df['cluster'].isin(large_cells))

count_df = count_df[spe_extract_6 | spe_extract_25]

# Export the counts per date per expert
mean_std = pd.DataFrame()
for d, group in count_df.groupby(['date']):
    name = list(group['date'])[0]
    data = group[['acq', 'cluster', 'Count', 'expert']]
    data = pd.pivot_table(data, values='Count', index=['cluster'], columns=['expert'], aggfunc=np.sum)
    data = data.fillna(0)
    data = data.astype(int)
    data.to_csv(temp_dir + 'SSLAMM_' + name + '.csv') 

    mean_std_acq = pd.concat([data.mean(1), data.std(1)], axis = 1)
    mean_std_acq.columns = ['mean', 'std']
    mean_std_acq = mean_std_acq['mean'].round(2).astype(str) + \
        ' (' + mean_std_acq['std'].round(2).astype(str) + ')'
    mean_std_acq.name = d   
    
    mean_std = pd.concat([mean_std, mean_std_acq], axis = 1)
    
    
mean_std.to_csv(temp_dir + 'SSLAMM_meanstd' + '.csv') 
mean_std.to_latex(temp_dir + 'SSLAMM_meanstd' + '.tex')

#======================================================================
# Check counts computed by Cytoclus == counts from Pulse shapes
#====================================================================== 
# Be careful to mean when value does not exist for some experts
# Conversion dict is bugged see new SECTIIION

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
        #counts = homogeneous_cluster_names(counts)
        counts = counts.groupby(['cluster']).sum()['Count'].reset_index()
        
        # Add the missing classes 
        clusters_in_counts = list(counts['cluster'])
        clusters_in_counts = [re.sub(' ', '', cc) for cc in clusters_in_counts]
        for cc in cluster_classes:
            # [re.search(cc_regex, list(counts['cluster']))]
            if not(cc in clusters_in_counts):
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
                except ValueError:
                    try:
                        file = pd.read_csv(pulse_dirs + '/' + expert + \
                         '/' + file_name, sep = ',', dtype = np.float64)
                    except (pd.errors.EmptyDataError):
                        print('Empty dataset')
                        continue

                                    
            # 0 is used as particle separation sign in Pulse shapes
            file = file[np.sum(file, axis = 1) != 0] 
            
            try:
                ids = np.unique(file['Particle ID'])
            except KeyError:
                ids = np.unique(file['ID'])

            cc_count= len(ids)
            
            pulse_counts = pulse_counts.append({'cluster': cluster,\
                            'Count': cc_count}, ignore_index=True)

        #pulse_counts = homogeneous_cluster_names(pulse_counts)
        pulse_counts = pulse_counts.groupby(['cluster']).sum().reset_index()
        
        # Add the missing classes 
        for cc in cluster_classes:
            if not(cc in clusters_in_counts):
                pulse_counts = pulse_counts.append({'cluster': cc, 'Count': 0},\
                                       ignore_index=True)
        
        counts_comp = counts.merge(pulse_counts, on = 'cluster')
        counts_dif = np.abs(counts_comp['Count_x'] - counts_comp['Count_y']).sum()
        
        if counts_dif:
            print("error")
            print(counts_comp)
            raise RuntimeError('Count difference')
            print('--------------------------------------')
        

#==============================================================================
# Create "unbiased" datasets from all manual classifications (voting)
#==============================================================================

# Exist duplicated PIDS due to Latimier non exclusive sets
# Noise work for previous section but not this one !!!

unbiased_part = pd.DataFrame(columns = ['Particle ID', 'cluster', 'acq'])

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
        
        # Need small post-processing for Creach's name
        if expert == 'Creach':
            ccs_expert2 = creach_post_processing(acq, ccs_expert)
        else:
            ccs_expert2 = ccs_expert
        
        ccs_expert_homogeneous = homogeneous_cluster_names(ccs_expert2)
        
        # Need to reverse the dict for Creach...
        names_corr = pd.DataFrame(zip(ccs_expert_homogeneous, ccs_expert),\
                                  columns = ['New', 'Old'])
        
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
                        
            if cluster not in set(names_corr.New):
                continue
                
            if cluster == 'Default (all)': # Useless ?
                    continue
            
            # Extract all the files corresponding to the current cluster
            file_names_cc = list(names_corr[names_corr['New'] == cluster]['Old'])
            
            for f in file_names_cc:
                file_name = acq + '_' + f + '_Pulses.csv'
                            
                try:
                    file = pd.read_csv(pulse_dirs + '/' + expert + \
                             '/' + file_name, sep = ';', dtype = np.float64)
                except ValueError: # If the data are in European format ("," stands for decimals and not thousands)
                    try:
                        file = pd.read_csv(pulse_dirs + '/' + expert + \
                             '/' + file_name, sep = ';', dtype = np.float64,\
                            thousands='.', decimal=',')
                    except ValueError:
                        try:
                            file = pd.read_csv(pulse_dirs + '/' + expert + \
                             '/' + file_name, sep = ',', dtype = np.float64)
                        except pd.errors.EmptyDataError:
                            print('Empty dataset')
                            continue
                except FileNotFoundError:
                    print(cluster, 'does not exists')
                    continue
                                            
                # 0 is used as particle separation sign in Pulse shapes
                file = file[np.sum(file, axis = 1) != 0] 
                
                if 'Particle ID' not in file.columns:
                    file = file.rename(columns = {'ID': 'Particle ID'})
                
                pid_cluster = deepcopy(pd.DataFrame(file['Particle ID']))
                pid_cluster['cluster'] = cluster   
                pid_cluster = pid_cluster.drop_duplicates()
                pid_clusters = pid_clusters.append(pid_cluster)

        #********************************************************************
        # Add the noise particles of Louchart
        #********************************************************************
         
        # To check  
        if expert == 'Louchart': 
            file_name_noise = [f for f in pulses_files_acq if re.search('unidentified', f)][0]
            file = pd.read_csv(pulse_dirs + 'Louchart/' + file_name_noise,\
                                sep = ';', dtype = np.float64, thousands='.',\
                                decimal=',')
                
            file = file[np.sum(file, axis = 1) != 0] 
            
            # If total FWS > 100 then noise particles are big noise particle
            total_fws = file[['Particle ID', 'FWS']].groupby(['Particle ID']).agg(trapz)
            big_noise_bool = list((total_fws >= 100).FWS)
            
            sup1um_noise_idx = total_fws.index[big_noise_bool]
            sup1um_noise = pd.DataFrame(data = sup1um_noise_idx, columns = ['Particle ID'])
            sup1um_noise['cluster'] = 'sup1microm_unidentified_particle'
            
            inf1um_noise_idx  = total_fws.index[np.logical_not(big_noise_bool)]
            inf1um_noise = pd.DataFrame(data = inf1um_noise_idx, columns = ['Particle ID'])
            inf1um_noise['cluster'] = 'inf1microm_unidentified_particle'
            
            pid_clusters = pid_clusters.append(inf1um_noise)
            pid_clusters = pid_clusters.append(sup1um_noise)
                
        #********************************************************************
        # Count how many times the particles have been counted in each class
        #********************************************************************
        dummy_pid_clusters = pd.get_dummies(pid_clusters.set_index('Particle ID'))
        
        #a, b = np.unique(pid_clusters['Particle ID'], return_counts = True) 
        #print('Number of duplicates', len(a[b == 2]))
        #assert len(a[b == 2]) == 0
        
        all_expert_pid_clusters = all_expert_pid_clusters.add(dummy_pid_clusters,\
                                                              fill_value = 0)
            
        all_expert_pid_clusters = all_expert_pid_clusters.fillna(0)
       
    #********************************************************************
    # Keep the particles that have been similarly 
    # classified by a majority/all experts
    #********************************************************************
    
    # Not all experts vote for each particle 
    nb_voting_experts_per_pid = all_expert_pid_clusters.sum(1)
    
    #================
    # Analyse the voting patterns
    #==============
    
    plt.hist(list(nb_voting_experts_per_pid))
    plt.title(acq)
    #plt.savefig('unbiased/' + acq)
    plt.show()
    #print(np.unique(list(nb_voting_experts_per_pid), return_counts = True))
    
    # Analyse which group with low votes
    prefered_group = all_expert_pid_clusters.idxmax("columns", skipna = False)
    low_votes = nb_voting_experts_per_pid <= 1
    # Often little noise
    
    assert len(low_votes) == len(prefered_group) 
    
    print(np.unique(prefered_group.loc[low_votes], return_counts = True))
    
    ################################
    
    maj_voting_idx = all_expert_pid_clusters.ge(nb_voting_experts_per_pid * (2/3), axis = 0).sum(1) == 1

    all_expert_pid_clusters.columns = [re.sub('cluster_', '', \
                                              col) for col in all_expert_pid_clusters]
    
    # Format PID class for plotting uncertainty map
    pid_clus = pd.DataFrame(all_expert_pid_clusters.idxmax(1), columns = ['pred_class'])
    pid_clus['know_cclass'] = maj_voting_idx
    pid_clus.to_csv('unbiased/' + acq + '_selected.csv')
    
    # Delete the pids that are not supported by the experts
    total_lines = len(all_expert_pid_clusters)
    all_expert_pid_clusters = all_expert_pid_clusters.loc[maj_voting_idx]
    kept_lines = len(all_expert_pid_clusters)
    print('Percentage of lines kept: ', kept_lines/total_lines)
            
    df = all_expert_pid_clusters.idxmax(1).reset_index()
    df['acq'] = acq
    df.columns = ['Particle ID', 'cluster', 'acq']
            
    unbiased_part = unbiased_part.append(df)
        
   
unbiased_part.to_parquet('unbiased_particles_twothird.parq', compression = 'snappy',\
                         index = False)

ccount = np.unique(unbiased_part.cluster, return_counts = True)
dict(zip(ccount[0], ccount[1]))

#======================================================================
# Create Pulses and Listmodes files from consensual particles
#====================================================================== 
date_regex = "(20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:h|u)[0-9]{2})"

pf = fp.ParquetFile('unbiased_particles_twothird.parq')
part_to_keep = pf.to_pandas()

#*********************************
# From Pulses
#*********************************

dest_repo = 'C:/Users/rfuchs/Documents/cyto_classif'


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
pf = fp.ParquetFile('unbiased_particles_twothird.parq')
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




#======================================================================
# Plot uncertainty maps  
#====================================================================== 
# Pb to tackle before: !!! 
# Duplicated indices for selected file

unbias_repo = 'C:/Users/rfuchs/Documents/These/Oceano/XP_biais/unbiased'
nomenclature_path = 'C:/Users/rfuchs/Documents/cyto_classif/XP_Pulses_L2/train_test_nomenclature.csv'
fig_repo = 'C:/Users/rfuchs/Documents/These/Oceano/XP_biais/uncertainty_maps_unanimity/'

tn = pd.read_csv(nomenclature_path)
tn.columns = ['label', 'Particle_class']
tn = tn.append({'Particle_class': 9, 'label': 'unknown'}, ignore_index = True)

# Take the Default by Lotty (same for all expert)
default_repo = 'C:/Users/rfuchs/Documents/These/Oceano/XP_biais/Listmodes/' 

for acq in acquistion_names_lists:
    
    # Import selected file
    selected_file_name = 'unbiased/' + acq + '_selected.csv'
    selected_file = pd.read_csv(selected_file_name)
    
    # Duplicates due to Latimier
    original_len = len(selected_file)
    selected_file = selected_file.drop_duplicates(['Particle ID'])
    new_len = len(selected_file)
    
    # Set class to unknown when particles are not selected
    selected_file.iloc[:,1] = np.where(selected_file['know_cclass'], \
                                           selected_file['pred_class'],\
                                               'unknown')
        
    del(selected_file['know_cclass'])
    
    # Import default Pulse
    default_name = acq + '_Default (all)_Listmode.csv'
    default_acq = pd.read_csv(default_repo + default_name, sep = ';', decimal = ',') 
    
    full_file = default_acq.merge(selected_file, on = 'Particle ID', how = 'left')
    full_file['pred_class'] = full_file['pred_class'].fillna('unknown')
    
    full_file.set_index('Particle ID', inplace = True)
    q1 = 'FWS Total'
    q2 = 'FL Red Total'

    # Original function
    plot_2Dcyto(full_file, full_file['pred_class'], tn, q1, q2, str_labels = True, title = fig_repo + acq)



##############################################################################
# Preprocessing utilities
##############################################################################


#==============================================================================
# Create Count files from Pulses (if selection sets were not given)
#==============================================================================

expert = 'Latimier'
expert_Pulse_files_names = os.listdir(pulse_dirs + '/' + expert )

for acq in acquistion_names_lists:
    print(acq)
    #*******************************
    # Count from the Pulse shapes
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
    # Add missing cols
    pulse_counts['Filename'] = acq + '.cyz'
    pulse_counts['Concentration'] = np.nan
    pulse_counts = pulse_counts[['Filename', 'cluster', 'Count', 'Concentration']]
    pulse_counts.columns = ['Filename', 'Set', 'Count', 'Concentration']
    
    # Push it to the right directory (need to be first created manually)
    pulse_counts.to_csv(count_dirs + expert + '/' + acq + '.csv',\
                        sep = ';', index = False)
        
        
#================================================================================
# Rename the Pulse files so that they end with Pulses.csv
#================================================================================
path = r'C:/Users/rfuchs/Documents/These/Oceano/XP_biais/Pulse_shapes/Creach'
files_to_rename = os.listdir(path)

for file in files_to_rename[1:]:
    os.rename(path + '/' + file, path + '/' + file[9:-4] + '_Pulses.csv')
