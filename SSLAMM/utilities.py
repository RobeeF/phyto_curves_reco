# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 14:59:35 2021

@author: rfuchs
"""

import re
import numpy as np
import pandas as pd
from copy import deepcopy
import scipy.integrate as it 
from itertools import combinations
from scipy.signal import find_peaks
from sklearn.metrics import adjusted_rand_score


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

def ARI_matrix(df, filename):
    '''
    Compute the ARI between all possible combination of experts

    Parameters
    ----------
    df : TYPE
        The votes of each expert (column) for each particle (row).
    filename: str
        Name of the file

    Returns
    -------
    None.

    '''
    cols = df.columns.to_list()  
    
    # Compute all possible combinations
    expert_couples = combinations(cols, 2)
    aris = pd.DataFrame(columns = ['Expert1', 'Expert2', 'ARI'])
            
    for expert1, expert2 in expert_couples:
        ari = adjusted_rand_score(df[expert1], df[expert2])
        aris = aris.append({'Expert1': expert1, 'Expert2': expert2,\
                                'ARI': ari}, ignore_index = True)
    
    aris['filename'] = filename
    return aris

def split_noise(df_, thr, is_FlYellow = True):
    '''
    Split the noise between "small" and big "noise" particles 
    df (pd DataFrame): DataFrame containing Pulse data for all particles
    ---------------------------------------------------------------
    returns (pd DataFrame): The data with a splitted noise
    '''

    # Dirty: avoid to modify the original object
    df = deepcopy(df_)
     
    # To account for Orange Fluorescence       
    fl2 = 'Fl Yellow' if is_FlYellow else 'FL Orange'
    
    noise_cells = df[df['cluster'] == 'Unassigned Particles']
    noise_df = noise_cells.reset_index().groupby('Particle ID').agg(
    {
          'FWS': it.trapz,
          'SWS': it.trapz,
          fl2: it.trapz,
          'FL Red': it.trapz,
          'Curvature': it.trapz, 
          'cluster': lambda x: list(x)[0] # Fetch the name of the cluster for each particle   
    })
    
    small_idx = noise_df[noise_df['FWS'] <= thr].index  
    big_idx = noise_df[noise_df['FWS'] > thr].index  
        
    df.loc[df['Particle ID'].isin(small_idx), 'cluster'] = 'inf1microm'
    df.loc[df['Particle ID'].isin(big_idx), 'cluster'] = 'sup1microm'
    
    return df


def is_bell_curved(x):
    '''
    Check if a series of points is bell curved

    Parameters
    ----------
    x : pandas Series
        The curve to look for.

    Returns
    -------
    Bool
        True if bell curved False otherwise.

    '''
    prominence = 3.0
      
    peaks_id, _  = find_peaks(x, prominence = prominence)
    drops_id, _  = find_peaks(-x, prominence = prominence + 3.0)
          
    return (len(peaks_id) == 1) & (len(drops_id) <= 1)


def redpicopro_bell_curved(df):
    '''
    Only keeps bell-curved REDPICOPRO

    Parameters
    ----------
    df : pandas DataFrame
        Pulse Data.

    Returns
    -------
    pandas DataFrame
        The new datasets without the non-bell curved REDPICOPRO.

    '''

    redpicopro = df[df['cluster'] == 'REDPICOPRO']
    is_bell = redpicopro.groupby('Particle ID')['SWS'].apply(is_bell_curved)
    chosen_pids = is_bell[is_bell].index
    redpicopro = redpicopro[redpicopro['Particle ID'].isin(chosen_pids)]
    
    return df[df['cluster'] != 'REDPICOPRO'].append(redpicopro)


def select_particles(left_particles, left_labels, pfg, FWS_thrs, FLR_thrs,\
                     nb_particles_added = 1000, random_state=None, q2 = 'FL Red'):
    '''
    Select particles in a given FWS, FLR region

    Parameters
    ----------
    left_particles : TYPE
        DESCRIPTION.
    left_labels : TYPE
        DESCRIPTION.
    pfg : TYPE
        DESCRIPTION.
    FWS_thrs : TYPE
        DESCRIPTION.
    FLR_thrs : TYPE
        DESCRIPTION.
    nb_particles_added : TYPE, optional
        DESCRIPTION. The default is 1000.
    q2 str
    Returns
    -------
    sampled_particles : TYPE
        DESCRIPTION.
    sampled_labels : TYPE
        DESCRIPTION.

    '''

    suited = left_particles[(left_particles['FWS'].between(FWS_thrs[0], FWS_thrs[1])) &\
                            (left_particles[q2].between(FLR_thrs[0], FLR_thrs[1]))]
    suited_labels = left_labels.loc[suited.index] 
    sampled_labels = suited_labels[suited_labels == pfg].sample(nb_particles_added,\
                                                                random_state = random_state)
    sampled_particles = suited.loc[sampled_labels.index]
    return sampled_particles, sampled_labels

