# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:24:12 2019

@author: Robin
"""

from dataset_preprocessing import interp_sequences, homogeneous_cluster_names
import pandas as pd
import numpy as np
import scipy.integrate as it
from dataset_preprocessing import scaler
import fastparquet as fp
import re

def predict(source_path, dest_folder, model, tn, scale = False, is_ground_truth = True):
    ''' Predict the class of unlabelled data with a pre-trained model and store them in a folder
    source_path (str): The path to the file containing the formatted unlabeled data
    dest_folder (str): The folder to store the predictions
    model (ML model): the pre-trained model to use, in order to make predictions
    ----------------------------------------------------------------------------
    return (Nonetype): Write the results in a csv on hardisk directly 
        '''
    
    max_len = 120 # The standard length to which is sequence will be broadcasted

    pfile = fp.ParquetFile(source_path)
    df = pfile.to_pandas()
    
    try:
        df = df.set_index('Particle ID')
    except:
        print('Particle ID was not found in column names')
   
    
    if len(df) > 0: # Delete empty dataframes       
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
            
        grouped_df = df[['SWS','FWS', 'FL Orange', 'FL Red', 'Curvature']].groupby('Particle ID')
    
        total_df = grouped_df.agg(
        {
             'FWS':it.trapz,    # Sum duration per group
             'SWS': it.trapz,  # get the count of networks
             'FL Orange': it.trapz,
             'FL Red': it.trapz,
             'Curvature': it.trapz,
        })
        
        obs_list = [obs.values.T for pid, obs in grouped_df]
    
        # Sanity checks
        assert len(set(df.index)) == len(total_df)
        if is_ground_truth:
            assert len(set(df.index)) == len(true_labels)
            
        obs_list = interp_sequences(obs_list, max_len)
        
        X = np.transpose(obs_list, (0, 2, 1))
        
        if scale:
            X = scaler(X)
            
        preds = np.argmax(model.predict(X), axis = 1)
        
        
        if is_ground_truth:
            formatted_preds = pd.DataFrame({'Particle ID': pid_list, \
                                            'Total FWS': total_df['FWS'], 'Total SWS': total_df['SWS'], \
                                            'Total FLO': total_df['FL Orange'], 'Total FLR': total_df['FL Red'], \
                                            'Total CURV': total_df['Curvature'], \
                                            'True FFT id': None, 'True FFT Label': true_labels, \
                                            'Pred FFT id': preds, 'Pred FFT Label': None}) 
    
        else:
            formatted_preds = pd.DataFrame({'Particle ID': pid_list, \
                                            'Total FWS': total_df['FWS'], 'Total SWS': total_df['SWS'], \
                                            'Total FLO': total_df['FL Orange'], 'Total FLR': total_df['FL Red'], \
                                            'Total CURV': total_df['Curvature'], \
                                            'Pred FFT id': preds, 'Pred FFT Label': None})         
        # Add string labels
        tn_dict = tn.set_index('label')['Particle_class'].to_dict()
    
        for id_, label in tn_dict.items():
            formatted_preds.loc[formatted_preds['Pred FFT id'] == id_, 'Pred FFT Label'] = label
    
            if is_ground_truth:
                formatted_preds.loc[formatted_preds['True FFT Label'] == label, 'True FFT id'] = id_
                    
    
        # Store the predictions on hard disk 
        date_regex = "(Pulse[0-9]{1,2}_20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:u|h)[0-9]{2})"
        file_name = re.search(date_regex, source_path).group(1)
        formatted_preds.to_csv(dest_folder + '/' + file_name + '.csv', index = False)
        
        
    else:
        print('File was empty.')


