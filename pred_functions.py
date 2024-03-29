# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:24:12 2019

@author: Robin
"""

import re
import numpy as np
import pandas as pd
import fastparquet as fp
import scipy.integrate as it
from dataset_preprocessing import interp_sequences, homogeneous_cluster_names

# Can be adapted (ex for SWINGS)
spe_extract = {'ORGNANO': 6, 'ORGPICOPRO': 25, 'REDNANO': 6, 'REDPICOEUK': 6,\
             'REDPICOPRO': 25, 'MICRO': 25, 'inf1microm': 6, 'sup1microm': 25}


def format_data(source_path, dest_folder, fluo2 = 'FL Orange',\
                is_ground_truth = True, hard_store = False):
    ''' Integrate the curves data
    source_path (str): The path to the file containing the formatted unlabeled data
    dest_folder (str): The folder to store the predictions
    is_ground_truth (Bool): Has the data been manually labelled ?
    Hard_store (Bool): Whether to return the data or store them on hardDisk
    ----------------------------------------------------------------------------
    return (Nonetype): Write the results in a np compressed format on hardisk directly 
    '''
    max_len = 120 # The standard length to which is sequence will be broadcasted

    pfile = fp.ParquetFile(source_path)
    df = pfile.to_pandas()
    
    try: # Dirty try / except to remove
        df = df.set_index('Particle ID')
    except:
        if 'ID' in df.columns: # Handle cytoclus3 formatting
            df = df.rename(columns={'ID': 'Particle ID'})
        else:
            print('Particle ID was not found in column names')
   
    #==========================================
    # Delete duplicates and integrate the curves
    #==========================================
    
    if len(df) > 0:       
        if is_ground_truth:
            df = homogeneous_cluster_names(df)
            true_labels = df.groupby('Particle ID')['cluster'].apply(np.unique)
            
            # Delete particles affiliated to 2 different groups
            not_corrupted_idx = true_labels.loc[true_labels.apply(len) == 1].index
            df = df.loc[not_corrupted_idx]
            pid_list = list(not_corrupted_idx)
            
            true_labels = true_labels.loc[not_corrupted_idx]
            true_labels = np.stack(true_labels)[:,0]
            
        else:   
            pid_list = list(set(df.index))
            
        grouped_df = df[['FWS', 'SWS', fluo2, 'FL Red', 'Curvature']].groupby('Particle ID')
    
        total_df = grouped_df.agg(
        {
             'FWS':it.trapz,    # Sum duration per group
             'SWS': it.trapz,  # get the count of networks
              fluo2: it.trapz,
             'FL Red': it.trapz,
             'Curvature': it.trapz,
        })
        
        obs_list = [obs.values.T for pid, obs in grouped_df]
    
        #==========================================
        # Sanity checks
        #==========================================

        assert len(set(df.index)) == len(total_df)
        if is_ground_truth:
            assert len(set(df.index)) == len(true_labels)
            
        obs_list = interp_sequences(obs_list, max_len)
        
        X = np.transpose(obs_list, (0, 2, 1))
        
        
        #==========================================
        # Store X, total_df, pid_list, true_labels in the same dir
        #==========================================
        
        if hard_store:
            file_name = re.search('/([A-Za-z0-9_\- ]+).parq', source_path).group(1)
                        
            np.savez_compressed(dest_folder + '/' +  file_name + '_X.npz', X)
            
            total_df.to_parquet(dest_folder + '/' + file_name + '_total_df.parq', compression = 'snappy') 
            np.savez_compressed(dest_folder + '/' + file_name + '_pid_list.npz', pid_list)

            if is_ground_truth:
                np.savez(dest_folder + '/'+ file_name + '_true_labels.npz', true_labels)

        else:
            if is_ground_truth:
              return X, total_df, pid_list, true_labels
            else:
              return X, total_df, pid_list, []
    
    else:
        return [], [], [], []
        

def predict(source_path, dest_folder, model, tn, fluo2 = 'FL Orange',\
            is_ground_truth = True):
    ''' Predict the class of unlabelled data with a pre-trained model and store them in a folder
    source_path (str): The path to the file containing the formatted unlabeled data
    dest_folder (str): The folder to store the predictions
    model (ML model): the pre-trained model to use, in order to make predictions
    is_ground_truth (Bool): Has the data been manually labelled ?
    ----------------------------------------------------------------------------
    return (Nonetype): Write the results in a csv on hardisk directly 
    '''
    date_regex = "(Pulse[0-9]{1,2}_20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:u|h)[0-9]{2})"
   
        
    #==========================================
    # Format the data and predict their class
    #==========================================
        
    X, total_df, pid_list, true_labels = format_data(source_path, dest_folder, fluo2,\
                    is_ground_truth = is_ground_truth, hard_store = False)

    if len(X) > 0:
        preds_probas = model.predict(X)
        preds = np.argmax(preds_probas, axis = 1)
        
        if is_ground_truth:
            formatted_preds = pd.DataFrame({'Particle ID': pid_list, \
                                            'Total FWS': total_df['FWS'], 'Total SWS': total_df['SWS'], \
                                            'Total FLO': total_df[fluo2], 'Total FLR': total_df['FL Red'], \
                                            'Total CURV': total_df['Curvature'], \
                                            'True PFG id': None, 'True PFG name': true_labels, \
                                            'Pred PFG id': preds, 'Pred PFG name': None,\
                                            'Pred PFG proba': preds_probas.max(1)}) 
    
        else:
            formatted_preds = pd.DataFrame({'Particle ID': pid_list, \
                                            'Total FWS': total_df['FWS'], 'Total SWS': total_df['SWS'], \
                                            'Total FLO': total_df[fluo2], 'Total FLR': total_df['FL Red'], \
                                            'Total CURV': total_df['Curvature'], \
                                            'Pred PFG id': preds, 'Pred PFG name': None,\
                                            'Pred PFG proba': preds_probas.max(1)}) 
        
        #==========================================
        # Add string labels
        #==========================================

        tn_dict = tn.set_index('id')['name'].to_dict()

        for id_, label in tn_dict.items():
            formatted_preds.loc[formatted_preds['Pred PFG id'] == id_, 'Pred PFG name'] = label
    
            if is_ground_truth:
                formatted_preds.loc[formatted_preds['True PFG name'] == label, 'True PFG id'] = id_
                    
        #==========================================
        # Store the predictions on hard disk 
        #==========================================

        file_name = re.search(date_regex, source_path).group(1)
        fp.write(dest_folder + '/' + file_name + '.parq',\
                     formatted_preds, compression='SNAPPY', write_index = False)
        
    else:
        print('File was empty.')


def combine_files_into_acquisitions(df):
    '''
    Merge the predictions made for the FLR[low threshold] file and FLR[high threshold]
    for each acquisition based on the spe_extract dict and delete the "corrupted" files

    Parameters
    ----------
    df : pandas DataFrame
        A dataFrame containing the abundances for each acqusition date and FLR.
    spe_extract : dict
        A dict that associates each PFG to a FLR threshold.
        The abundance of this PFG are fetched in the chosen FLR-threshold files.

    Returns
    -------
    df_rpz_ts : pandas DataFrame
        The compiled abundances

    '''

    #===========================================================================  
    # Set to zero the unused PFG abundances from the other FLR acquisitions
    #===========================================================================  
    
    for pfg, flr in spe_extract.items():
        df.loc[df['FLR'] != flr, pfg] = 0
    
    #===========================================================================  
    # Deal with acquisitions that have several FLR6 or FLR25 for the same date
    #===========================================================================  

    idx_pbs = pd.DataFrame(df.groupby(['date', 'FLR']).size()) 
    idx_pbs = idx_pbs[idx_pbs[0] > 1].index
    idx_pbs = [id_[0] for id_ in  idx_pbs] # Fetch the problematic file dates
    
    df_ok = df[~df['date'].isin(idx_pbs)]
    
    # Take the more likely entry:
    df_resolved_pbs =  df[df['date'].isin(idx_pbs)].groupby(['date', 'FLR']).max().reset_index() 
    df = pd.concat([df_ok, df_resolved_pbs]).reset_index(drop = True)
    
    #===========================================================================  
    # Deal with acquisitions which have a FLR6 but no FLR25 (or the reverse)
    #===========================================================================  
    
    idx_pbs = pd.DataFrame(df.groupby('date').size())  
    idx_pbs = idx_pbs[idx_pbs[0] == 1].index
    
    df_rpz_ts = df.set_index('date').astype(float).reset_index()
    df_rpz_ts = df_rpz_ts.groupby('date').sum().reset_index()
    
    # For those which have not both a FLR[low] and a FLR[high] file, replace the missing values by NaN
    for idx in idx_pbs:
        df_rpz_ts[df_rpz_ts['date'] == idx] = df_rpz_ts[df_rpz_ts['date'] == idx].replace(0, np.nan)
        
    del(df_rpz_ts['FLR'])
    return df_rpz_ts
