# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 11:33:52 2020

@author: rfuchs
"""

import os
import re
import clr
import numpy as np
import pandas as pd
import seaborn as sns
import fastparquet as fp
import scipy.integrate as it
import matplotlib.pyplot as plt

# CytoDAL utilities
documentsDirectory=os.path.expanduser(r'~\Documents')
cytoDALpath=os.path.join(documentsDirectory,r'CytoFuchs2\CytoDAL.dll')
cyzDALpath=os.path.join(documentsDirectory,r'CytoFuchs2\CytoSense.dll')

clr.setPreload(True)
clr.AddReference(cytoDALpath)
clr.AddReference(cyzDALpath)
clr.AddReference('System.Runtime')
clr.AddReference('System.Runtime.InteropServices')

import CytoDAL as cd 


code_dir = 'C:/Users/rfuchs/Documents/GitHub/'
XP_dir = r'C:/Users/rfuchs/Documents/These/Oceano/XP_bias_SWINGS/'

os.chdir(code_dir)

from SWINGS.utilities import nonconsensualPicoNano, split_noise, create_cyto_file,\
    redpicopro_bell_curved, ARI_matrix
from phyto_curves_reco.dataset_preprocessing import homogeneous_cluster_names_swings
from phyto_curves_reco.viz_functions import plot_2Dcyto


# Regular expressions:
flr_regex = 'FLR([0-9]{1,2})'
datetime_regex = "(20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}h[0-9]{2})"

fileDirectory= r'C:/Users/rfuchs/Documents/These/Oceano/XP_bias_SWINGS/files_to_class'
filenames = [f for f in os.listdir(fileDirectory) if f.endswith('cyz')]
datetimes = [re.search(datetime_regex, f).group(1) for f in filenames]

experts = os.listdir(r'C:/Users/rfuchs/Documents/These/Oceano/XP_bias_SWINGS/selectionSets')
experts.sort()
n_experts = len(experts)

pulse_dir = XP_dir + 'Pulse_shapes/'
parq_dir = XP_dir + 'L1/'
consensus_dir = 'consensual_PIDs/'
listmode_dir = 'Listmodes/'

cluster_classes = ['ORGPICOPRO', 'MICRO', 'REDNANO', 'ORGNANO', 
  'REDPICOEUK', 'REDPICOPRO' , 'Unassigned Particles']

consensus_ratio =  2/3
CVs = pd.DataFrame()


colors = {'ORGPICOPRO' : 'darkorange', 'REDPICOEUK': 'tomato', 'REDNANO': 'maroon', 'REDPICOPRO': 'red',\
          'REDMICRO' : 'darkblue', 'inf1microm': 'chocolate', 'sup1microm': 'peachpuff',\
          'Non-consensual' : 'black', 'ORGNANO': 'goldenrod', 'MICRO' : 'mediumblue',\
          'Unassigned Particles' : 'lightgreen'}

# To split between "small" and "big" noise
noise_thr = 1E3    

#===============================================
# Label non-consensual particles and compute CVs
#===============================================

aris = pd.DataFrame()

for fileName in filenames:
    # Create the .cyto copy of the .cyz file
    cyzFilePath = os.path.join(fileDirectory, fileName) 
    cytoFilePath = create_cyto_file(cyzFilePath) 
    cytoFilePath = cyzFilePath.replace('cyz', 'cyto')
    cytoFile = cd.CytoFile(cytoFilePath)

    # Get the number of particles and the datetime of the acquisition
    n_particles = cytoFile.ParticleCount
    datetime = re.search(datetime_regex, fileName).group(1) 
    
    df = pd.DataFrame(columns = experts, index = cytoFile.ParticleIDs)

    #===============================================
    # Get the particle IDs and their class 
    #===============================================
    
    for expert in experts:
        # Fetch the corresponding setFile
        expert_sets_dir = os.path.join(XP_dir, 'selectionSets', expert)
        setNames = os.listdir(expert_sets_dir)
        
        # Fetch the manual gates of each class
        try:
            setName = [set_ for set_ in setNames if re.search(datetime, set_)][0]
            class_gates = cytoFile.LoadSetList(os.path.join(expert_sets_dir, setName)) 
        except:
            print('expert', expert, 'has not labelled', fileName)
            del(df[expert])
            continue
            

        # Fetch the particles and normalise their cluster name
        for gate in class_gates:      
            gateName = homogeneous_cluster_names_swings([gate.Name])[0]
            if gateName in cluster_classes:
                indices = [idx for idx in gate.ParticleIndices]
                df[expert][indices] = gateName  
        
    #===============================================
    # Cluster counts, means and standard error 
    #===============================================
    
    # NaN particles are also considered "Unassigned"
    df = df.fillna('Unassigned Particles')
    
    # Compute the ARI
    aris = aris.append(ARI_matrix(df, fileName))
    
    # Count the number of votes per PFG 
    votes = df.apply(pd.Series.value_counts, axis=1).fillna(0) 
    # Count the number of experts that have classsified this file
    experts_file = df.columns
    n_experts_file = len(experts_file)
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
    
    # Compute the variation coefficients 
    counts = pd.concat([df[expert].value_counts() for expert in experts_file], axis = 1) 
    counts = counts.fillna(0)
    CVs_file = counts.std(1) / counts.mean(1)
    CVs_file = pd.DataFrame(CVs_file, columns = [fileName]).T
    CVs = CVs.append(CVs_file)
    
    print(fileName, 'had', n_experts_file, 'experts')
    particle_class.to_csv(XP_dir + consensus_dir + fileName[:-4] + '.csv')
  
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
    pulseName = fileName[:-4] + '_Pulses.csv'
    pulseFile = pd.read_csv(pulse_dir + pulseName, sep = ';', decimal = ',')
    
    # 0 is used as particle separation sign in Pulse shapes
    pulseFile = pulseFile[np.sum(pulseFile, axis = 1) != 0] 
    
    # Label the Pulse data
    consensualPulse = pulseFile.merge(consensual[['Particle ID', 'cluster']], how = 'inner')  
    assert len(consensualPulse) == len(pulseFile) # Sanity check  
    
    # Assign to non-consensual non-consensual particles at the PICO/ NANO frontier
    consensualPulse = nonconsensualPicoNano(consensualPulse)
    
    # Correct the consensual file from these unassigned PICO / NANO particles
    corrected_consensual = consensualPulse[['Particle ID', 'cluster']].drop_duplicates().reset_index(drop = True)
    corrected_consensual.to_csv(XP_dir + consensus_dir + fileName[:-4] + '.csv', index = False)
    
    # Delete non bell-curved Prochlorococcus
    if pd.Series(['REDPICOPRO']).isin(consensualPulse['cluster']).values[0]:
        print('There are some REDPICOPROS')

    print(consensualPulse.shape)
    consensualPulse = redpicopro_bell_curved(consensualPulse)
    print(consensualPulse.shape)
    
    # Delete non-consensual particles
    consensualPulse = consensualPulse[consensualPulse['cluster'] != 'Non-consensual']

    # Split the noise between small and big noise
    consensualPulse = split_noise(consensualPulse, thr = noise_thr)
    consensualPulse = consensualPulse.reset_index(drop = True)
    
    # Write the consensual particles
    parqName = fileName[:-4] + '.parq'
    fp.write(parq_dir + 'Pulse/' + parqName, consensualPulse, compression='SNAPPY',\
             write_index = False)     
  
#****************************
# Listmodes
#****************************

for fileName in filenames:
    print(fileName)
    
    # Open Pulse file and keep the manual gating labels  
    pulseName = fileName[:-4] + '.parq' 
    pfile = fp.ParquetFile(parq_dir + 'Pulse/' + pulseName)
    pids_labels = pfile.to_pandas(columns = ['Particle ID', 'cluster'])
    #pids_labels = pids_labels.reset_index().drop_duplicates()

    pids_labels = pids_labels.drop_duplicates()
    
    # Open the .parq file
    lmName = fileName[:-4] + '_Default (all)_Listmode.csv'
    lmFile = pd.read_csv(XP_dir + listmode_dir + lmName, sep = ';', decimal = ',')
        
    # Label the Pulse data (and delete the non-consensual particles during the merge)
    consensualLm = lmFile.merge(pids_labels, how = 'right')  
    assert len(consensualLm) == len(pids_labels) # Sanity check  
    
    # Write Consensual Pulse files
    parqName = fileName[:-4] + '.parq'
    fp.write(parq_dir + 'Listmodes/' +  parqName, consensualLm, compression='SNAPPY',\
             write_index = False)     
  

#===============================================
# Plot the uncertainty maps
#===============================================

for fileName in filenames:
    # Open consensual file
    consensual = pd.read_csv(XP_dir + consensus_dir + fileName[:-4] + '.csv')
    
    # Open the .parq file
    pulseName = fileName[:-4] + '_Pulses.csv'
    pulseFile = pd.read_csv(pulse_dir + pulseName, sep = ';', decimal = ',')
    
    # 0 is used as particle separation sign in Pulse shapes
    pulseFile = pulseFile[np.sum(pulseFile, axis = 1) != 0] 
    
    # Label the Pulse data
    consensualPulse = pulseFile.merge(consensual[['Particle ID', 'cluster']], how = 'inner')
    # Old Sanity check : now that some REDPICOPROs are withdrawn:  len(consensualPulse) < len(pulseFile)  
    #assert len(consensualPulse) == len(pulseFile)  
    
    # Compute the integrated signal
    consensualIt = consensualPulse.reset_index().groupby('Particle ID').agg(
    {
          'FWS': it.trapz,
          'SWS': it.trapz,
          'Fl Yellow': it.trapz,
          'FL Red': it.trapz,
          'Curvature': it.trapz, 
          'cluster': lambda x: list(x)[0] # Fetch the name of the cluster for each particle   
    })
    
    # Plot the integrated signal
    imgName = fileName[:-4] + '.png'
    q1, q2 = 'FL Red', 'Fl Yellow'
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

CVs = CVs[classes]
formatted_CVs = pd.melt(CVs, value_vars = CVs.columns, var_name='PFG', value_name='CV')

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

            

