# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 11:33:52 2020

@author: rfuchs
"""

import os 
import re
import numpy as np
import pandas as pd
import seaborn as sns
import fastparquet as fp
import matplotlib.pyplot as plt
import scipy.integrate as it


os.chdir('C:/Users/rfuchs/Documents/GitHub/phyto_curves_reco')
from dataset_preprocessing import homogeneous_cluster_names
from viz_functions import plot_2Dcyto

os.chdir('C:/Users/rfuchs/Documents/GitHub/SSLAMM')
from utilities import creach_post_processing, ARI_matrix, split_noise, redpicopro_bell_curved


os.chdir('C:/Users/rfuchs/Documents/These/Oceano/XP_bias_SSLAMM')

count_dirs = 'Counts/'
pulse_dirs = 'Pulse_shapes_experts/'

experts = os.listdir(count_dirs)
n_experts = len(experts)
consensus_ratio = 2/3 
CVs = pd.DataFrame()

filenames = os.listdir(count_dirs + 'Lotty')

enriched_files = ['SSLAMM-FLR6 2019-12-08 15h59',
 'SSLAMM-FLR6 2020-02-19 05h59',
 'SSLAMM-FLR6 2020-06-26 11h59',
 'SSLAMM_FLR25 2019-12-08 16h07',
 'SSLAMM_FLR25 2020-02-19 06h07',
 'SSLAMM_FLR25 2020-06-26 12h07']


# Regular expressions:
flr_regex = 'FLR([0-9]{1,2})'
date_regex = "(20[0-9]{2}-[0-9]{2}-[0-9]{2})"

cc_regex = '[0-9]_([_\-() 0-9A-Za-zµ]+)_Pulses.csv'
date_regex = "(20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:h|u)[0-9]{2})"

 
cluster_classes = ['ORGPICOPRO', 'MICRO', 'REDNANO', 'ORGNANO', 
  'REDPICOEUK', 'REDPICOPRO' ,'Unassigned Particles']
cluster_classes.sort()


# Transform the name of the classes
new_nomenclature = {'synechococcus':'ORGPICOPRO', 'microphytoplancton':'MICRO',\
                    'nanoeucaryote': 'REDNANO', 'cryptophyte': 'ORGNANO', 
                    'picoeucaryote': 'REDPICOEUK', 'prochlorococcus': 'REDPICOPRO',\
                    'Unassigned Particles': 'Unassigned Particles'}

    
XP_dir = 'C:/Users/rfuchs/Documents/These/Oceano/XP_bias_SSLAMM/'
consensus_dir = 'consensual_PIDs/'
pulse_dir = XP_dir + 'Pulse_shapes_experts/'
parq_dir = XP_dir + 'L1/'
consensus_dir = 'consensual_PIDs/'
listmode_dir = 'Listmodes/'
default_pulse_dir = XP_dir + 'Pulse_shapes/'


colors = {'ORGPICOPRO' : 'darkorange', 'REDPICOEUK': 'tomato', 'REDNANO': 'maroon', 'REDPICOPRO': 'red',\
          'REDMICRO' : 'darkblue', 'inf1microm': 'chocolate', 'sup1microm': 'peachpuff',\
          'Non-consensual' : 'black', 'ORGNANO': 'goldenrod', 'MICRO' : 'mediumblue',\
          'Unassigned Particles' : 'lightgreen'}
    
noise_thr = 1E2

#===============================================
# Label non-consensual particles and compute CVs
#===============================================
aris = pd.DataFrame()

for acqName in filenames:
    
    df = pd.DataFrame(columns = experts)
    date = re.search(date_regex, acqName).group(1) 
    
    #===============================================
    # Get the particle IDs and their class 
    #===============================================
    
    for expert in experts:
        pid_clusters = pd.DataFrame()
        #******************************* 
        # Open Pulse files
        #*******************************
        
        pulseFiles = os.listdir(pulse_dirs + '/' + expert)
        pulseFiles = [name for name in pulseFiles if re.search(date, name)]
                                
        for pulseFile in pulseFiles:
            
            #****************************************
            # Format the name of the PFG
            #****************************************

            cluster = re.search(cc_regex, pulseFile).group(1) 
            
            if cluster == 'Default (all)':
                continue

            if expert == 'Creach':
               cluster = creach_post_processing(acqName, [cluster])[0]
               
            cluster = homogeneous_cluster_names([cluster])[0]
            
            # Convert unknown particles as unidentified
            if cluster not in set(new_nomenclature.keys()):
                cluster = 'Unassigned Particles'
            else:
                cluster = new_nomenclature[cluster]
                        
            #****************************************
            # Get the particle IDs and their class 
            #****************************************
            
            file = pd.read_csv(pulse_dirs + '/' + expert + \
                             '/' + pulseFile, sep = ';', dtype = np.float64).iloc[:, 1:]
             
            
            # 0 is used as particle separation sign in Pulse shapes
            file = file[np.sum(file, axis = 1) != 0] 
            
            if 'Particle ID' not in file.columns:
                file = file.rename(columns = {'ID': 'Particle ID'})
            
            pid_cluster = file[['Particle ID']].drop_duplicates()
            pid_cluster['cluster'] = cluster   
            pid_clusters = pid_clusters.append(pid_cluster)
            
            
        # Store the PIDs and the clusters
        pid_clusters = pid_clusters.set_index('Particle ID').sort_index()
        # Drop the duplicates of Latimier
        pid_clusters = pid_clusters[~pid_clusters.index.duplicated(keep='first')]
        df[expert] = pid_clusters['cluster']
    
    df = df.fillna('Unassigned Particles')
    
    # Compute the ARI
    aris = aris.append(ARI_matrix(df, acqName))
        
    #===============================================
    # Cluster counts, means and standard error 
    #===============================================
    
    # Count the number of votes per PFG 
    votes = df.apply(pd.Series.value_counts, axis=1).fillna(0)
    # Count the number of experts that have classsified this file
    n_experts_file = df.shape[1]
    # Get consensual Particle IDs
    voted_idx = (votes / n_experts_file >= consensus_ratio).any(1)
    # Get the class of the consensual particles
    voted_particles = votes.loc[voted_idx].T.idxmax()
    
    # Get the PID of the non-consensual particles
    unvoted_idx = df[~voted_idx].index 
    unvoted_particles = pd.Series(['Non-consensual'] * len(unvoted_idx), index = unvoted_idx)
    
    # Pull together consensual and non-consensual particles
    particle_class = voted_particles.append(unvoted_particles).sort_index()
    particle_class = pd.DataFrame(particle_class, columns = ['cluster'])
    particle_class.index.name = 'Particle ID'

    assert len(df) == len(particle_class)
    print(acqName, 'had', n_experts_file, 'experts')
    particle_class.to_csv(XP_dir + consensus_dir + acqName)
    
    # Compute the variation coefficients 
    counts = pd.concat([df[expert].value_counts() for expert in experts], axis = 1) 
    counts = counts.fillna(0)
    CVs_file = counts.std(1) / counts.mean(1)
    CVs_file = pd.DataFrame(CVs_file, columns = [acqName]).T
    CVs = CVs.append(CVs_file)
    
CVs.to_csv(XP_dir + 'CVs.csv')  
aris.to_csv(XP_dir + 'ARI.csv', index = False)  


#===============================================
# Create the consensual .parq files
#===============================================

#****************************
# Pulse shapes
#****************************

for fileName in filenames:
    print(fileName) 
    # Open consensual file
    consensual = pd.read_csv(XP_dir + consensus_dir + fileName[:-4] + '.csv')
    
    # Open the Pulse file
    pulseName = fileName[:-4] + '_Default (all)_Pulses.csv'
    pulseFile = pd.read_csv(default_pulse_dir + pulseName, sep = ';',\
                            dtype = np.float).iloc[:, 1:]
    
    # Label the Pulse data
    consensualPulse = pulseFile.merge(consensual[['Particle ID', 'cluster']], how = 'inner')  
    assert len(consensualPulse) == len(pulseFile) # Sanity check  

    # Delete non-Bell curved Prochlorococcus
    if pd.Series(['REDPICOPRO']).isin(consensualPulse['cluster']).values[0]:
        print('There are some REDPICOPROS')
    
    o_shape = len(set(consensualPulse['Particle ID']))
    consensualPulse = redpicopro_bell_curved(consensualPulse)
    n_shape = len(set(consensualPulse['Particle ID']))
    print(o_shape - n_shape)
    print(consensualPulse.shape)
    
    # Correct the consensual file from these unassigned PICO / NANO particles
    #corrected_consensual = consensualPulse[['Particle ID', 'cluster']].drop_duplicates().reset_index(drop = True)
    #corrected_consensual.to_csv(XP_dir + consensus_dir + fileName[:-4] + '.csv', index = False)
    
    # Delete non-consensual particles
    consensualPulse = consensualPulse[consensualPulse['cluster'] != 'Non-consensual']
    
    # Split the noise between small and big noise
    consensualPulse = split_noise(consensualPulse, thr = noise_thr, is_FlYellow = False)
    
    parqName = fileName[:-4] + '.parq'
    fp.write(parq_dir + 'Pulse/' + parqName, consensualPulse, compression='SNAPPY')     
  
    
#****************************
# Listmodes
#****************************

for fileName in filenames:
    print(fileName)
    
    # Open Pulse file and keep the manual gating labels  
    pulseName = fileName[:-4] + '.parq' 
    pfile = fp.ParquetFile(parq_dir + 'Pulse/' + pulseName)
    pids_labels = pfile.to_pandas(columns = ['Particle ID', 'cluster'])
    pids_labels = pids_labels.drop_duplicates()
    
    # Open the .parq file
    lmName = fileName[:-4] + '_Default (all)_Listmode.csv'
    lmFile = pd.read_csv(XP_dir + listmode_dir + lmName, sep = ';', decimal = ',')
        
    # Label the Pulse data (and delete the non-consensual particles during the merge)
    consensualLm = lmFile.merge(pids_labels, how = 'right')  
    assert len(consensualLm) == len(pids_labels) # Sanity check  
    
    # Write Consensual Pulse files
    parqName = fileName[:-4] + '.parq'
    fp.write(parq_dir + 'Listmodes/' +  parqName, consensualLm, compression='SNAPPY',   
             write_index = False)     
  

#===============================================
# Plot the uncertainty maps
#===============================================

for fileName in filenames:
    # Open consensual file
    consensual = pd.read_csv(XP_dir + consensus_dir + fileName[:-4] + '.csv')
    
    # Open the .parq file
    pulseName = fileName[:-4] + '_Default (all)_Pulses.csv'
    pulseFile = pd.read_csv(default_pulse_dir + pulseName,\
                            sep = ';', dtype = np.float).iloc[:, 1:]
    
    # 0 is used as particle separation sign in Pulse shapes
    pulseFile = pulseFile[np.sum(pulseFile, axis = 1) != 0] 
    
    # Label the Pulse data
    consensualPulse = pulseFile.merge(consensual[['Particle ID', 'cluster']], how = 'inner')
    # Sanity check    
    assert len(consensualPulse) == len(pulseFile)  
    
    # Compute the integrated signal
    consensualIt = consensualPulse.reset_index().groupby('Particle ID').agg(
    {
          'FWS': it.trapz,
          'SWS': it.trapz,
          'FL Orange': it.trapz,
          'FL Red': it.trapz,
          'Curvature': it.trapz, 
          'cluster': lambda x: list(x)[0] # Fetch the name of the cluster for each particle   
    })
    
    # Plot the integrated signal
    imgName = fileName[:-4] + '.png'
    q1, q2 = 'FL Red', 'FL Orange'
    X, y = consensualIt.iloc[:,:5], consensualIt.iloc[:,-1]
    plot_2Dcyto(X, y, q1, q2, colors = colors,\
                title = XP_dir + '/uncertainty_maps/' +\
                q1 + '_' + q2 + '_' + imgName)
    
#===============================================
# Plot the boxplots
#===============================================

CVs  = pd.read_csv(XP_dir + 'CVs.csv').iloc[:,1:] 

# Deal with the absence of REDPICOPRO
classes = ['ORGPICOPRO', 'MICRO', 'REDNANO', 'ORGNANO', 'REDPICOEUK', 'REDPICOPRO']
classes.sort()
formatted_CVs = pd.melt(CVs, value_vars= CVs.columns, var_name='PFG', value_name='CV')

# Deal with the absence of REDPICOPRO

ax = sns.boxplot(x = "PFG", y = "CV", 
                 data = formatted_CVs, palette="Set3",\
                     showfliers = False, order  = classes)
ax.tick_params('x', rotation = 30)
plt.ylim([-0.1,2.6])
plt.tight_layout()
plt.savefig(XP_dir + 'CVs.png')


#=============================
# ARI coefficient
#=============================

df = pd.read_csv(XP_dir + 'ARI.csv')

q1 = np.quantile(df['ARI'], q = 0.25)
q3 = np.quantile(df['ARI'], q = 0.75)

plt.axvline(q1, color = 'tab:orange')
plt.axvline(q3, color = 'tab:orange')

plt.hist(df['ARI'], bins = 30)
plt.xlim([0,1])
plt.xlabel('ARI value')
plt.ylabel('Number of pairwise ARIs')
plt.title('Adjusted Rand Index distribution')
