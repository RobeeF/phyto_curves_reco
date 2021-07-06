# -*- coding: utf-8 -*-
"""
Created on Fri May 14 16:55:01 2021

@author: rfuchs
"""

import os 
import re
import numpy as np
import pandas as pd
import seaborn as sns
from copy import deepcopy
import fastparquet as fp
import matplotlib.pyplot as plt
from scipy.integrate import trapz


os.chdir('C:/Users/rfuchs/Documents/GitHub/phyto_curves_reco')
from dataset_preprocessing import homogeneous_cluster_names_swings
from viz_functions import plot_2Dcyto

# Path to change
os.chdir('C:/Users/rfuchs/Documents/These/Oceano/XP_biais')

count_dirs = 'Counts/'
pulse_dirs = 'Pulse_shapes/'

expert_names_list = os.listdir(count_dirs)
acquistion_names_lists = []


# Regular expressions:
flr_regex = 'FLR([0-9]{1,2})'
date_regex = "(20[0-9]{2}-[0-9]{2}-[0-9]{2})"

expert = expert_names_list[0]
acq = acquistion_names_lists[0]
cc_regex = '[0-9]_([_\-() 0-9A-Za-zµ]+)_Pulses.csv'

cluster_classes = [] # To fill

classes_of_interest = [] # If all classes are of interest set equal to cluster_classes 

small_cells = [] # PFG counted on FLR5 files
large_cells = [] # PFG counted on FLR20 files


#======================================================================
# Compute average counts and std (biases) over manual classifications
#====================================================================== 

temp_dir = 'C:/Users/rfuchs/Documents/These/Oceano/XP_bias_SWINGS/count/'

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
        

        counts = homogeneous_cluster_names_swings(counts)        
        counts = counts.groupby(['cluster']).sum().reset_index()
        
        # Add the missing classes 
        for cc in cluster_classes:
            if not(cc in list(counts['cluster'])):
                counts = counts.append({'cluster': cc, 'Count': 0},\
                                       ignore_index=True)
                    
        # Report expert and acquisition names
        counts['expert'] = expert
        counts['acq'] = acq
        
        count_df = count_df.append(counts)

print(np.unique(count_df["cluster"]))
count_df['Count'] = count_df['Count'].astype(int)  

#==============================================
# Compute the total phyto particles identified 
#==============================================
              
tot_phyto = count_df[count_df['cluster'].isin(classes_of_interest)][['acq', 'Count',\
                    "expert"]].groupby(['acq', "expert"]).sum()
                 
tot_part = count_df[count_df['cluster'] == 'default(all)'][['acq', 'Count',  "expert"]].groupby(['acq', "expert"]).sum()


# Add the noise counts per experts
noise_count = (tot_part - tot_phyto).reset_index()
noise_count['cluster'] = 'Noise'
noise_count['Concentration'] = np.nan
noise_count = noise_count[count_df.columns]

# Delete detailed noise particles from the count
count_df = count_df[count_df['cluster'].isin(classes_of_interest)]
count_df = count_df.append(noise_count)

count_df.groupby(['acq', 'expert']).sum()['Count'].groupby('acq').std()


#==============================================
# Merge the FLR5 and FLR20 files of the same day
#==============================================

count_df['flr'] = count_df['acq'].str.extract(flr_regex).astype(int)
count_df['date'] = count_df['acq'].str.extract(date_regex)

c = count_df.set_index(['acq', 'expert']).sort_index().reset_index()

c.groupby(['date', 'expert']).sum()

# Export the counts per date per expert
# The noise is here aggregated (inf + sup 1 microm) but a detail can be obtained by running the
# code starting on line 857 
mean_std = pd.DataFrame()
for d, group in count_df.groupby(['date']):
    name = list(group['date'])[0]
    data = group[['acq', 'cluster', 'Count', 'expert']]
    data = pd.pivot_table(data, values='Count', index=['cluster'], columns=['expert'], aggfunc=np.sum)
    data = data.fillna(0)
    data = data.astype(int)
    data.to_csv(temp_dir + 'SSLAMM_' + name + '.csv') 

    print(data.std(1) / data.mean(1))
    mean_std_acq = pd.concat([data.mean(1), data.std(1)], axis = 1)
    mean_std_acq.columns = ['mean', 'std']
    mean_std_acq = mean_std_acq['mean'].round(2).astype(str) + \
        ' (' + mean_std_acq['std'].round(2).astype(str) + ')'
    mean_std_acq.name = d   
    
    mean_std = pd.concat([mean_std, mean_std_acq], axis = 1)
    
    
mean_std.to_csv(temp_dir + 'SSLAMM_meanstd_noise' + '.csv') 
mean_std.to_latex(temp_dir + 'SSLAMM_meanstd_noise' + '.tex')


#==============================================
# Plotting the groups per acquisition
#==============================================

# Graphes pour Creach
for date_idx, date_data in count_df.groupby('date'):
    sns.boxplot(y='Count', x='cluster', 
                 data=date_data, 
                 palette="colorblind",
                 order =['airbubble', 'cryptophyte', 'nanoeucaryote',\
                'microphytoplancton',\
                'picoeucaryote', 'prochlorococcus', \
                'synechococcus'],
                 #hue='Gating type', 
                 showfliers = False
                 )
    
    plt.title('Acquisition date: ' + date_idx)
    locs, labels=plt.xticks()
    plt.xticks(locs, ['airbubbles', 'cryptophytes', 'nanoeukaryotes',\
                'microphytoplankton', 'picoeukaryotes', 'Prochlorococcus', \
                'Synechococcus'], fontsize = 10, rotation=20)
        
    plt.xlabel(xlabel = 'Phytoplankton Functional Groups', fontsize = 10)
    plt.ylabel(ylabel = 'Counts (number of particles)', fontsize = 10)

    plt.savefig('C:/Users/rfuchs/Desktop/manual_gating_diversity/' + date_idx + '.png',\
                bbox_inches='tight')   
    plt.show()



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
# Create "unbiased" datasets from all manual classifications (2/3 voting rule)
#==============================================================================

# Exist duplicated PIDS due to Latimier non exclusive sets

unbiased_part = pd.DataFrame(columns = ['Particle ID', 'cluster', 'acq'])

for acq in acquistion_names_lists:
    print(acq)
    
    all_expert_pid_clusters = pd.DataFrame()
    
    # Create a empty dataframe with the right columns and particle number 
    for expert in expert_names_list:
        print('Expert:', expert)

        #*******************************
        # Build a correspondance dict
        #******************************* 
        
        # To link the names of the files and the Original class
        expert_Pulse_files_names = os.listdir(pulse_dirs + '/' + expert)
        
        ccs_expert = [re.search(cc_regex, name).group(1) for name in expert_Pulse_files_names]
        ccs_expert = list(set(ccs_expert))
                
        ccs_expert_homogeneous = homogeneous_cluster_names_swings(ccs_expert)
        
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
                    #print(cluster, 'does not exists')
                    continue
                                            
                # 0 is used as particle separation sign in Pulse shapes
                file = file[np.sum(file, axis = 1) != 0] 
                
                if 'Particle ID' not in file.columns:
                    file = file.rename(columns = {'ID': 'Particle ID'})
                
                pid_cluster = deepcopy(pd.DataFrame(file['Particle ID']))
                pid_cluster['cluster'] = cluster   
                pid_cluster = pid_cluster.drop_duplicates()
                
                # Plusieurs classes pour une particule ? 
                #assert len(pid_cluster) == len(set(pid_cluster['Particle ID']))
                
                pid_clusters = pid_clusters.append(pid_cluster)


                
        #********************************************************************
        # Count how many times the particles have been counted in each class
        #********************************************************************
        dummy_pid_clusters = pd.get_dummies(pid_clusters.set_index('Particle ID'))
        # Re sum to account for Latimier duplicates
        dummy_pid_clusters = dummy_pid_clusters.reset_index('Particle ID').groupby('Particle ID').agg('sum')
        
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
    print('Number of particle in file', len(all_expert_pid_clusters))
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
    
    #print(np.unique(prefered_group.loc[low_votes], return_counts = True))
    
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

part_to_keep = part_to_keep.drop_duplicates()

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
    unbiased_parts = file.merge(acq_part[['Particle ID', 'cluster']], how = 'inner')
    
    print('Keep ', len(set(unbiased_parts['Particle ID'])), 'over', len(set(file['Particle ID'])), 'particles = ',\
          len(set(unbiased_parts['Particle ID'])) / len(set(file['Particle ID'])), '%')

    fp.write(dest_repo + '/XP_Pulses_L2' + '/Labelled_Pulse' + str(flr_num) + '_' +\
             date + '.parq', unbiased_parts, compression='SNAPPY')

#*********************************
# From Listmodes
#*********************************

print('--------------------------------------------')
os.chdir('C:/Users/rfuchs/Documents/These/Oceano/XP_biais')

list_dir = 'Listmodes'
pf = fp.ParquetFile('unbiased_particles_twothird.parq')
part_to_keep = pf.to_pandas()

part_to_keep = part_to_keep.drop_duplicates()


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
        
    print(len(file.columns))
        
    # Label the unbiased data
    unbiased_parts = file.merge(acq_part[['Particle ID', 'cluster']], how = 'inner')
    
    print('Keep ', len(unbiased_parts), 'over', len(file), 'particles = ',\
          len(unbiased_parts) / len(file), '%')

    fp.write(dest_repo + '/XP_Listmodes' + '/Labelled_Pulse' + str(flr_num) + '_' +\
             date + '.parq', unbiased_parts, compression='SNAPPY')
    

len(set(unbiased_parts_['Particle ID']))
len(set(unbiased_parts['Particle ID']))

ccl, count = np.unique(unbiased_parts['Particle ID'], return_counts = True)
len(ccl[count >= 2])

file[file['Particle ID'] == 4643]
acq_part[acq_part['Particle ID'] == 4643]['cluster']

#======================================================================
# Plot uncertainty maps  
#====================================================================== 

unbias_repo = 'C:/Users/rfuchs/Documents/These/Oceano/XP_biais/unbiased'
nomenclature_path = 'C:/Users/rfuchs/Documents/cyto_classif/XP_Pulses_L2/train_test_nomenclature.csv'
fig_repo = 'C:/Users/rfuchs/Documents/These/Oceano/XP_biais/uncertainty_maps_twothird/'

tn = pd.read_csv(nomenclature_path)
tn.columns = ['label', 'Particle_class']
tn = tn.append({'Particle_class': 9, 'label': 'non-consensual particles'}, ignore_index = True)


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
                                               'non-consensual particles')
        
    del(selected_file['know_cclass'])
    
    # Import default Pulse
    default_name = acq + '_Default (all)_Listmode.csv'
    default_acq = pd.read_csv(default_repo + default_name, sep = ';', decimal = ',') 
    
    full_file = default_acq.merge(selected_file, on = 'Particle ID', how = 'left')
    full_file['pred_class'] = full_file['pred_class'].fillna('non-consensual particles')
    
    full_file.set_index('Particle ID', inplace = True)
    q2 = 'FL Red Total'
    q1 = 'SWS Total'

    # Original function
    plot_2Dcyto(full_file, full_file['pred_class'], tn, q1, q2, str_labels = True,\
                title = fig_repo + acq + '_' + q1 + '_' + q2)
    #plot_2Dcyto(full_file, full_file['pred_class'], tn, q1, q2, str_labels = True)



##############################################################################################
###########  Compute actual noise heterogenity computation) ############################
##############################################################################################

import re
os.chdir('C:/Users/rfuchs/Documents/GitHub/phyto_curves_reco')
from dataset_preprocessing import homogeneous_cluster_names
import scipy.integrate as it

parent_repo = 'C:/Users/rfuchs/Documents/These/Oceano/XP_biais/Pulse_shapes/'
expert_repos = os.listdir(parent_repo)
flr_num = 6

is_ground_truth = True
source = 'C:/Users/rfuchs/Desktop/formatted_XP/'
dest_folder = 'C:/Users/rfuchs/Desktop/noise_estimate/' 


date_regex = "(20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}h[0-9]{2})"
flr_regex = 'Pulse([0-9]{1,2})'

inf_noise_count_df = pd.DataFrame(index = acquistion_names_lists, columns = expert_names_list)
sup_noise_count_df = pd.DataFrame(index = acquistion_names_lists, columns = expert_names_list)

for expert in expert_repos:
    print(expert)
    
    files = os.listdir(source + expert)
    files = [f for f in files if re.search('.parq', f)]
    
    for file in files: 
        print(file)
        source_path = source + expert + '/' + file
        
        pfile = fp.ParquetFile(source_path)
        df = pfile.to_pandas()
        
        try:
            df = df.set_index('Particle ID')
        except:
            if 'ID' in df.columns: # Handle cytoclus3 formatting
                df = df.rename(columns={'ID': 'Particle ID'})
            else:
                print('Particle ID was not found in column names')
   
        grouped_df = df[['SWS', 'FWS', 'cluster']].reset_index().groupby(['Particle ID', 'cluster'])
    
        df = grouped_df.agg(
        {
             'FWS':it.trapz,    # Sum duration per group
        })
        df.reset_index(inplace = True)
            
        # Need small post-processing for Creach's name
        if expert == 'Creach':
            df = creach_post_processing(acq, df)

        df = homogeneous_cluster_names(df)
        true_labels = df.groupby('Particle ID')['cluster'].apply(np.unique)
        df = df.set_index('Particle ID')
   
        # Delete particles affiliated to 2 different groups
        if len(true_labels) != len(df):

            not_corrupted_idx = true_labels.loc[true_labels.apply(len) == 1].index
            #df.reindex(index = not_corrupted_idx)
            df = df.loc[not_corrupted_idx]
            pid_list = list(not_corrupted_idx)
            
            true_labels = true_labels.loc[not_corrupted_idx]
            
        true_labels = np.stack(true_labels)[:,0]
        print(set(true_labels))
        
        true_labels = np.where((true_labels == 'noise') & (df['FWS'] >= 100), 'sup1microm_unidentified_particle', true_labels)
        true_labels = np.where((true_labels == 'noise') & (df['FWS'] < 100), 'inf1microm_unidentified_particle', true_labels)
        
        cnames, ccounts = np.unique(true_labels, return_counts = True)
        inf_noise = ccounts[cnames == 'inf1microm_unidentified_particle'][0]
        sup_noise = ccounts[cnames == 'sup1microm_unidentified_particle'][0]
        
        # Need to extract the date before
        date = re.search(date_regex, file).group(1)
        flr_num = re.search(flr_regex, file).group(1)
        if flr_num == '25':
            formatted_name = 'SSLAMM_FLR' + flr_num + ' ' + date
        else:
            formatted_name = 'SSLAMM-FLR' + flr_num + ' ' + date

        inf_noise_count_df.loc[formatted_name, expert] = inf_noise
        sup_noise_count_df.loc[formatted_name, expert] = sup_noise
   
        
#np.unique(true_labels, return_counts = True)

temp_dir = 'C:/Users/rfuchs/Documents/These/Oceano/XP_biais/temp_count3/'

inf_noise_count_df.to_csv(temp_dir + 'inf1microm_noise.csv')  
sup_noise_count_df.to_csv(temp_dir + 'sup1microm_noise.csv')  


# Extract the mean and noise information
mean_std_acq = pd.concat([inf_noise_count_df.mean(axis = 1), inf_noise_count_df.std(axis = 1) ], axis = 1)
mean_std_acq.columns = ['mean', 'std']
print(mean_std_acq['std'] / mean_std_acq['mean'])

mean_std_acq = mean_std_acq['mean'].round(2).astype(str) + \
    ' (' + mean_std_acq['std'].round(2).astype(str) + ')'
mean_std_acq.index.name = 'date'
mean_std_acq.name = 'mean_std'
mean_std_acq = mean_std_acq.reset_index().T

mean_std_acq.columns = mean_std_acq.iloc[0]
mean_std_acq = mean_std_acq[1:]
mean_std_acq.to_csv(temp_dir + 'inf1microm_noise_agg.csv')  


# Same with sup1 microm noise
mean_std_acq = pd.concat([sup_noise_count_df.mean(axis = 1), sup_noise_count_df.std(axis = 1) ], axis = 1)
mean_std_acq.columns = ['mean', 'std']
print(mean_std_acq['std'] / mean_std_acq['mean'])

mean_std_acq = mean_std_acq['mean'].round(2).astype(str) + \
    ' (' + mean_std_acq['std'].round(2).astype(str) + ')'
mean_std_acq.index.name = 'date'
mean_std_acq.name = 'mean_std'
mean_std_acq = mean_std_acq.reset_index().T

mean_std_acq.columns = mean_std_acq.iloc[0]
mean_std_acq = mean_std_acq[1:]
mean_std_acq.to_csv(temp_dir + 'sup1microm_noise_agg.csv')  
