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
from collections import Counter
from copy import deepcopy


def format_data(source_path, dest_folder, scale = False, \
                is_ground_truth = True, hard_store = False):
    max_len = 120 # The standard length to which is sequence will be broadcasted

    pfile = fp.ParquetFile(source_path)
    df = pfile.to_pandas()
    
    try:
        df = df.set_index('Particle ID')
    except:
        if 'ID' in df.columns: # Handle cytoclus3 formatting
            df = df.rename(columns={'ID': 'Particle ID'})
        else:
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
            
        if hard_store:
            # Store X, total_df, pid_list, true_labels in the same dir
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
        

def predict(source_path, dest_folder, model, tn, scale = False,\
            is_ground_truth = True, precomputed_data_dir = ''):
    ''' Predict the class of unlabelled data with a pre-trained model and store them in a folder
    source_path (str): The path to the file containing the formatted unlabeled data
    dest_folder (str): The folder to store the predictions
    model (ML model): the pre-trained model to use, in order to make predictions
    precomputed_data_dir (str): A folder where the quantities have been precomputed 
        using format_data with the hard_store argument
    ----------------------------------------------------------------------------
    return (Nonetype): Write the results in a csv on hardisk directly 
    '''
    
    is_precomputed_data = (precomputed_data_dir != '')
    
    if is_precomputed_data:
        file_name = re.search('/([A-Za-z0-9_\- ]+).parq', source_path).group(1)
            
        X = np.load(precomputed_data_dir + '/' +  file_name + '_X.npz')['arr_0']
        
        pfile = fp.ParquetFile(precomputed_data_dir + '/' + file_name + '_total_df.parq')
        total_df = pfile.to_pandas()
            
        pid_list = np.load(precomputed_data_dir + '/' + file_name + '_pid_list.npz')['arr_0']
        
        if is_ground_truth:
            true_labels = np.load(dest_folder + '/'+ file_name + '_true_labels.npz')['arr_0']
        
    else:
        X, total_df, pid_list, true_labels = format_data(source_path, dest_folder, scale = scale, \
                        is_ground_truth = is_ground_truth, hard_store = False)
            
    if len(X) > 0:
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
        #fp.write(dest_folder + '/' + file_name + '.parq', formatted_preds, compression='SNAPPY')
        
    else:
        print('File was empty.')


def correc_pred_thr(tn, thrs, pred_proba):
    '''When the model is not very confident, classify the observation as noise '''
    preds = pred_proba.argmax(1)
    very_confident_pred = deepcopy(preds)
    
    max_pred_proba = pred_proba.max(1)
    
    for cl in range(len(tn)):
      preds_cl_mask = (preds == cl)
      not_sure_mask = max_pred_proba < thrs[cl]
      very_confident_pred[not_sure_mask & preds_cl_mask] = 7
    
      return very_confident_pred


def pred_n_count(source_path, model, tn, thrs, exp_count = False):
    ''' Predict the observations and count on the fly '''
    max_len = 120 # The standard length to which is sequence will be broadcasted
    
    #================================================
    # Import and format data
    #================================================
    
    pfile = fp.ParquetFile(source_path)
    df = pfile.to_pandas()
    df = df.set_index('Particle ID')
               
    grouped_df = df[['SWS','FWS', 'FL Orange', 'FL Red', 'Curvature']].groupby('Particle ID')

    obs_list = [obs.values.T for pid, obs in grouped_df]        
    obs_list = interp_sequences(obs_list, max_len)

    X = np.transpose(obs_list, (0, 2, 1))
    
    #================================================
    # Predict and format predictions
    #================================================
    preds_oh = model.predict(X)
    
    #preds_proba = preds_oh.sum(0).round().astype(int)
    #keys = range(len(tn))
    #count_proba = {key: preds_proba[key] for key in keys}
    
    preds = correc_pred_thr(tn, thrs, preds_oh)
    count = dict(Counter(preds))

    lab_count = {}
    #lab_count_proba = {}
    
    for i in range(len(tn)):
        try:
            lab_count[list(tn[tn['label'] == i]['Particle_class'])[0]] = count[i]
            #lab_count_proba[list(tn[tn['label'] == i]['Particle_class'])[0]] = count_proba[i]
        except KeyError:
            print('key number ', i, 'was not found')
   
    cl_count = pd.DataFrame(pd.Series(lab_count)).T
    #cl_count_proba = pd.DataFrame(pd.Series(lab_count_proba)).T
    
    #================================================
    # Keep only interesting particles
    #================================================
    flr_regex = 'Pulse([0-9]{1,2})'

    flr_num = int(re.search(flr_regex, source_path).group(1))
    
    # Keep only "big" phytoplancton from FLR25 and "small" one from FLR6 
    if flr_num == 25:
        for clus_name in ['picoeucaryote', 'synechococcus', 'prochlorococcus']:
            if clus_name in cl_count.columns:
                cl_count[clus_name] = 0
                #cl_count_proba[clus_name] = 0

        
    elif flr_num == 6:
        for clus_name in ['cryptophyte', 'nanoeucaryote', 'microphytoplancton']:
            if clus_name in cl_count.columns:
                cl_count[clus_name] = 0
                #cl_count_proba[clus_name] = 0

    else:
        raise RuntimeError('Unkonwn flr number', flr_num)
    
    
    #======================================================
    # Format the date
    #======================================================
    
    # The timestamp is rounded to the closest 20 minutes 
    date_regex = "Pulse[0-9]{1,2}_(20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:u|h)[0-9]{2})"

    date = re.search(date_regex, source_path).group(1)
    date = pd.to_datetime(date, format='%Y-%m-%d %Hh%M', errors='ignore')
    
    date = date.round('1H')
    '''
    mins = date.minute
                
    if (mins >= 00) & (mins < 15): 
        date = date.replace(minute=00)

    elif (mins >= 15) & (mins <= 35): 
        date = date.replace(minute=20)
    
    elif (mins > 35) & (mins < 57):
        date = date.replace(minute=40)
        
    elif mins >= 57:
        if date.hour != 23:
            date = date.replace(hour= date.hour + 1, minute=00)
        else:
            try:
                date = date.replace(day = date.day + 1, hour = 00, minute=00)
            except:
                date = date.replace(month = date.month + 1, day = 1, hour = 00, minute=00)
    '''
    cl_count['date'] = date 
    #cl_count_proba['date'] = date 


    return cl_count#, cl_count_proba


def post_processing(df, flr_num):
    ''' 
        Reassign some of the particles classified by the CNN into the right classes
        df (pandas DataFrame): The CNN predictions
        flr_num (int): The FLR threshold of the predictions
        ---------------------------------------------------------------
        returns (pandas DataFrame): The corrected CNN predictions
    '''
    SWS_noise_thr = 70
    FWS_crypto_thr = 1E4
    FWS_micros_thr = 2 * 10 ** 5
    
    if flr_num == 25:
        df.loc[np.logical_and(df['Total FWS'] <= FWS_crypto_thr, df['Pred FFT Label'] == 'cryptophyte'),\
                      'Pred FFT Label'] = 'sup1microm_unidentified_particle'
        df.loc[np.logical_and(df['Total FWS'] < FWS_micros_thr, df['Pred FFT Label'] == 'microphytoplancton'),\
                      'Pred FFT Label'] = 'nanoeucaryote'
    else:
        df.loc[((df['Total FWS'] <= 100) & (df['Total SWS'] >= SWS_noise_thr) \
               & (df['Pred FFT Label'] == 'inf1microm_unidentified_particle')),\
               'Pred FFT Label'] = 'prochlorococcus'
    return df


def combine_files_into_acquisitions(df):
    ''' Merge the predictions made for the FLR6 file and FLR25 for each 
        acquisition and delete the "corrupted" files
        df (pandas DataFrame): A dataFrame containing for each date (index) and 
        pfg (columns), a quantity such as the biomass, the biovolume or the Total FLR 
        ---------------------------------------------------------------
        returns (pandas DataFrame): The predicted quantity by date for each pfg
    '''
    
    #===========================================================================  
    # Deal with acquisitions that have several FLR6 or FLR25 for the same date
    #===========================================================================  

    idx_pbs = pd.DataFrame(df.groupby(['date', 'FLR']).size()) 
    idx_pbs = idx_pbs[idx_pbs[0] > 1].index
    idx_pbs = [id_[0] for id_ in  idx_pbs]
    
    df_ok = df[~df['date'].isin(idx_pbs)]
    df_resolved_pbs =  df[df['date'].isin(idx_pbs)].set_index(['date', 'FLR']).groupby(['date', 'FLR']).max().reset_index() # Take the more likely entry
    df = df_ok.reset_index(drop = True).append(df_resolved_pbs)
        
    #===========================================================================  
    # Deal with acquisitions which have a FLR6 but no FLR25 (or the reverse)
    #===========================================================================  
    
    idx_pbs = pd.DataFrame(df.groupby('date').size())  
    idx_pbs = idx_pbs[idx_pbs[0] == 1].index
    
    df_rpz_ts = df.set_index('date').astype(float).reset_index()
    df_rpz_ts = df_rpz_ts.groupby('date').sum()
    df_rpz_ts = df_rpz_ts.reset_index()
    
    # For those which have not both a FLR6 and a FLR25 file, replace the missing values by NaN
    for idx in idx_pbs:
        df_rpz_ts[df_rpz_ts['date'] == idx] = df_rpz_ts[df_rpz_ts['date'] == idx].replace(0, np.nan)
        
    del(df_rpz_ts['FLR'])
    return df_rpz_ts