# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 11:49:53 2021

@author: rfuchs
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 11:33:52 2020

@author: rfuchs
"""

import os
import clr
import ctypes
import numpy as np
import pandas as pd
import scipy.integrate as it
from scipy.stats import shapiro 
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import adjusted_rand_score



from copy import deepcopy

documentsDirectory=os.path.expanduser(r'~\Documents')

cytoDALpath=os.path.join(documentsDirectory,r'CytoFuchs2\CytoDAL.dll')
cyzDALpath=os.path.join(documentsDirectory,r'CytoFuchs2\CytoSense.dll')

clr.setPreload(True)
clr.AddReference(cytoDALpath)
clr.AddReference(cyzDALpath)
clr.AddReference('System.Runtime')
clr.AddReference('System.Runtime.InteropServices')

import CytoDAL as cd # contains .cyto support (read/write/calculations)
import CytoSense as cs # contains .cyz -> .cyto support

from System import Array, Int32
from System.Runtime.InteropServices import GCHandle, GCHandleType

# fast conversion from DotNet types
# https://github.com/pythonnet/pythonnet/issues/514

_MAP_NET_NP = {
    'Single' : np.dtype('float32'),
    'Double' : np.dtype('float64'),
    'SByte'  : np.dtype('int8'),
    'Int16'  : np.dtype('int16'), 
    'Int32'  : np.dtype('int32'),
    'Int64'  : np.dtype('int64'),
    'Byte'   : np.dtype('uint8'),
    'UInt16' : np.dtype('uint16'),
    'UInt32' : np.dtype('uint32'),
    'UInt64' : np.dtype('uint64'),
    'Boolean': np.dtype('bool'),
}

def DotNetToNumpyArray(netArray):
    '''
    Given a CLR `System.Array` returns a `numpy.ndarray`.  See _MAP_NET_NP for 
    the mapping of CLR types to Numpy dtypes.
    '''
    dims = np.empty(netArray.Rank, dtype=int)
    for I in range(netArray.Rank):
        dims[I] = netArray.GetLength(I)
    netType = netArray.GetType().GetElementType().Name

    try:
        npArray = np.empty(dims, order='C', dtype=_MAP_NET_NP[netType])
    except KeyError:
        raise NotImplementedError("asNumpyArray does not yet support System type {}".format(netType) )

    try: # Memmove 
        sourceHandle = GCHandle.Alloc(netArray, GCHandleType.Pinned)
        sourcePtr = sourceHandle.AddrOfPinnedObject().ToInt64()
        destPtr = npArray.__array_interface__['data'][0]
        ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
    finally:
        if sourceHandle.IsAllocated: sourceHandle.Free()
    return npArray


def create_cyto_file(filePath):
    """Creates a .cyto file from a .cyz file (preserves .cyz file)
    
    Note: Only converts if output file is not present, 
          stores output file in same directory as input file
    """
    
    outputPath=filePath.replace('.cyz','.cyto')

    if not os.path.exists(outputPath):
        print('Converting file: {}'.format(filePath))
        cyzConverter=cs.CyzConverter(filePath)
        #print(dir(cyzConverter))
       
        try:
            cyzConverter.ReadCyzFile()
            cyzConverter.ConvertToCyto()
        except:
            print('ERROR converting file: {}'.format(filePath))
            return ''
            
        cyzConverter.Dispose()
    return  outputPath


def plot_on_cytogram(all_x_values, all_y_values, color, particle_indices = None):
    """plot x,y values on a cytogram, use particle_indices to specify a subset
    
    Note: all_x_values, all_y_values and particle_indices MUST be numpy arrays.
    """
    
    if particle_indices is None:
        x_values=all_x_values
        y_values=all_y_values
    else: # select x,y values for specified particle_indices
        x_values=all_x_values[particle_indices]
        y_values=all_y_values[particle_indices]
    
    axis=plt.gca()
    axis.grid(True, which='both',linestyle='--')
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.plot(x_values, y_values, '.', c=color, alpha=0.4, markeredgecolor='none' )


def print_statistics(description, values):
    print("%-20s" % description,"  mean: %.2f" % values.mean(), " sdev(x): %.2f" % values.std()) 

def nonconsensualPicoNano(df):
    '''
    Assign to non-consensual non-consensual particles at the PICO/ NANO frontier

    Parameters
    ----------
    X : pd.DataFrame of array
        Covariates and dependant variables 

    Returns
    -------
    df : TYPE
        Clean version of the data.
    '''
    
    df_it = df.groupby('Particle ID').agg(
    {
          'FWS': it.trapz,
          'SWS': it.trapz,
          'Fl Yellow': it.trapz,
          'FL Red': it.trapz,
          'Curvature': it.trapz, 
          'cluster': lambda x: list(x)[0] # Fetch the name of the cluster for each particle   
    })
    
    nonConsensual_idx = df_it[(df_it['cluster'] == 'Unassigned Particles') \
                 & (df_it['FWS'] <= 1E5) \
                 & (df_it['FL Red'] >= 8 * 1E3)].index  
        
    df.loc[df['Particle ID'].isin(nonConsensual_idx), 'cluster'] = 'Non-consensual'
    
    return df

def split_noise(df_,thr, is_FlYellow = True):
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


def select_particles(left_particles, left_labels, pfg, FWS_thrs, FLR_thrs,\
                     nb_particles_added = 1000, random_state=None,  q1 = 'FWS',\
                     q2 = 'FL Red'):
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

    suited = left_particles[(left_particles[q1].between(FWS_thrs[0], FWS_thrs[1])) &\
                            (left_particles[q2].between(FLR_thrs[0], FLR_thrs[1]))]
    suited_labels = left_labels.loc[suited.index] 
    sampled_labels = suited_labels[suited_labels == pfg].sample(nb_particles_added,\
                                                                random_state = random_state)
    sampled_particles = suited.loc[sampled_labels.index]
    return sampled_particles, sampled_labels



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
