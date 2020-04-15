import pandas as pd
import os 
import re
import numpy as np
import fastparquet as fp

### Serious need for refactorisation here..

def extract_labeled_curves(data_source, data_destination, flr_num = 6, spe_extract_FLR = False):
    ''' Create 5 labelled curves per particle in the files in the data_source repository. 
    The curves data are placed in data_destination. This function works for the FUMSECK data
    data_source (str): The path to the original source of data
    data_features (str) : The path to write the data to
    flr_num (int): Either 6 or 25. Indicate whether the curves have to be generated from FLR6 or FLR25 files
    spe_extract_FLR (bool): Whether to extract only synecchocchus from FLR5
    ---------------------------------------------------------------------------------------------
    returns (None): Write the labelled Pulse shapes on hard disk
    '''
    assert (flr_num == 6) or (flr_num == 25)
    
    nb_files_already_processed = 0
    log_file = data_destination + "/pred_logs.txt" # Register where write the already predicted files
    if not(os.path.isfile(log_file)):
        open(data_destination + '/pred_logs.txt', 'w+').close()
    else:
        with open(data_destination + '/pred_logs.txt', 'r') as file: 
            nb_files_already_processed = len(file.readlines())

    
    files_title = [f for f in os.listdir(data_source)]
    # Keep only the interesting csv files
    flr_title = [f for f in files_title if re.search("FLR" + str(flr_num) + ' ',f) and re.search("csv",f) ]

    pulse_titles_clus = [f for f in flr_title if  re.search("Pulse",f) and not(re.search("Default",f))]
    pulse_titles_default = [f for f in flr_title if  re.search("Pulse",f) and re.search("Default",f)]

    # Defining the regex
    date_regex = "FLR" + str(flr_num) + " (20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}h[0-9]{2})_[A-Za-z ()]+"
    pulse_regex = "_([a-zA-Z0-9 ()]+)_Pulses.csv"  

    dates = set([re.search(date_regex, f).group(1) for f in flr_title if  re.search("Pulse",f)])
    cluster_classes = list(set([re.search(pulse_regex, cc).group(1) for cc in pulse_titles_clus]))
    
    if len(pulse_titles_default) != 0:
        cluster_classes += ['noise']
     
    nb_acquisitions = len(dates)
    
    if nb_acquisitions == 0:
        print('No file found...')
    
    #date = list(dates)[0]
    for date in dates: # For each acquisition
        print(nb_files_already_processed, '/', nb_acquisitions, "files have already been processed")
        print("Processing:", date)

        ### Check if the acquisition has already been formatted
        with open(log_file, "r") as file:
            if date + '_FLR' + str(flr_num) in file.read(): 
                print("Already formatted")
                continue
            
        pulse_data = pd.DataFrame()
        # Get the info about each particule and its cluster label
        date_datasets_titles = [t for t in pulse_titles_clus if re.search(date, t)]
        
        #title= date_datasets_titles[0]
        # For each file, i.e. for each functional group
        for title in date_datasets_titles:
            clus_name = re.search(pulse_regex, title).group(1) 
            
            try:
                df = pd.read_csv(data_source + '/' + title, sep = ';', dtype = np.float64)
            except ValueError: # If the data are in European format ("," stands for decimals and not thousands)
                try:
                    df = pd.read_csv(data_source + '/' + title, sep = ';', dtype = np.float64, thousands='.', decimal=',')
                except pd.errors.EmptyDataError:
                    print('Empty dataset')
                    continue

            df = df[df.values.sum(axis=1) != 0] # Delete formatting zeros
                                            
            # Add the date of the extraction
            df["date"] = date
            
            # Get the name of the cluster from the file name
            #clus_name = re.search(pulse_regex, title).group(1) 
            df["cluster"] = clus_name
            
            df.set_index("Particle ID", inplace = True)

            pulse_data = pulse_data.append(df)
        
        #===========================================================
        # Extract info of noise particles from the default file
        #===========================================================
        
        if len(pulse_titles_default) > 0:
            
            title = [t for t in pulse_titles_default if re.search(date, t)][0] # Dirty
            
            try:
                df = pd.read_csv(data_source + '/' + title, sep = ';', dtype = np.float64)
            except ValueError: # If the data are in European format ("," stands for decimals and not thousands)
                try:
                    df = pd.read_csv(data_source + '/' + title, sep = ';', dtype = np.float64, thousands='.', decimal=',')
                except pd.errors.EmptyDataError:
                    print('Empty dataset')
                    continue
            
            df = df[df.values.sum(axis=1) != 0] # Delete formatting zeros
            df["date"] = date
            
            existing_indices = pulse_data[pulse_data['date']==date].index
    
            df.set_index("Particle ID", inplace = True)
            noise_indices = list(set(df.index) - set(existing_indices)) # Determining the noise particles indices
            df = df.loc[noise_indices] # Keep only the noise particles 
        
            clus_name = "noise"
            df["cluster"] = clus_name
        
            pulse_data = pulse_data.append(df)

        if spe_extract_FLR & (flr_num == 25): 
            prev_len = len(pulse_data)
            unwanted_fft = ['picoeucaryotes', 'synechococcus', 'Prochlorococcus', 'PicoHIGHFLR']
            pulse_data = pulse_data[~pulse_data.cluster.isin(unwanted_fft)]
            new_len = len(pulse_data)
            print('Dropped', prev_len - new_len, 'lines')
            
        fp.write(data_destination + '/Labelled_Pulse' + str(flr_num) + '_' + date + '.parq', pulse_data, compression='SNAPPY')

    
        # Mark that the file has been formatted
        with open(log_file, "a") as file:
            file.write(date + '_FLR' + str(flr_num) + '\n')
            nb_files_already_processed += 1
        print('------------------------------------------------------------------------------------')


def extract_non_labeled_curves(data_source, data_destination, flr_num = 6):
    ''' Create 5 curves per particle in the files in the data_source repository. 
    The curves data are placed in data_destination. This function works for the FUMSECK data
    data_source (str): The path to the original source of data
    data_features (str) : The path to write the data to
    flr_num (int): Either 6 or 25. Indicate whether the curves have to be generated from FLR6 or FLR25 files
    spe_extract_FLR (bool): Whether to extract only synecchocchus from FLR5
    ---------------------------------------------------------------------------------------------
    returns (None): Write the labelled Pulse shapes on hard disk
    '''
    assert (flr_num == 6) or (flr_num == 25)
    
    nb_files_already_processed = 0
    log_file = data_destination + "/pred_logs.txt" # Register where write the already predicted files
    if not(os.path.isfile(log_file)):
        open(data_destination + '/pred_logs.txt', 'w+').close()
    else:
        with open(data_destination + '/pred_logs.txt', 'r') as file: 
            nb_files_already_processed = len(file.readlines())

    
    files_title = [f for f in os.listdir(data_source)]
    # Keep only the interesting csv files
    flr_title = [f for f in files_title if re.search("FLR" + str(flr_num) + ' ',f) and re.search("csv",f) ]

    # Defining the regex
    date_regex = "FLR" + str(flr_num) + " (20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}h[0-9]{2})_[A-Za-z ()]+"
    dates = set([re.search(date_regex, f).group(1) for f in flr_title if  re.search("Pulse",f)])
     
    nb_acquisitions = len(dates)
    
    if nb_acquisitions == 0:
        raise RuntimeError('No file found...')
    
    for date in dates: # For each acquisition
        print(nb_files_already_processed, '/', nb_acquisitions, "files have already been processed")
        print("Processing:", date)

        ### Check if the acquisition has already been formatted
        with open(log_file, "r") as file:
            if date + '_FLR' + str(flr_num) in file.read(): 
                print("Already formatted")
                continue
            
        ## Extract info of noise particles from the default file
        title = [t for t in flr_title if re.search(date, t)][0] # Dirty
        
        try:
            df = pd.read_csv(data_source + '/' + title, sep = ';', dtype = np.float64)
        except ValueError: # If the data are in European format ("," stands for decimals and not thousands)
            try:
                df = pd.read_csv(data_source + '/' + title, sep = ';', dtype = np.float64, thousands='.', decimal=',')
            except pd.errors.EmptyDataError:
                print('Empty dataset')
                continue
        
        df = df[df.values.sum(axis=1) != 0]        
        df["date"] = date
        
        fp.write(data_destination + '/Labelled_Pulse' + str(flr_num) + '_' + date + '.parq', df, compression='SNAPPY')

    
        # Mark that the file has been formatted
        with open(log_file, "a") as file:
            file.write(date + '_FLR' + str(flr_num) + '\n')
            nb_files_already_processed += 1
        print('------------------------------------------------------------------------------------')




        
# Need to be refactored with the previous function
def extract_Oscar(listmode_source, pulse_source, data_destination, flr_num = 5, is_untreated = True):
    ''' Get the label of each particle from the Listmodes of the OSCHAR campaign 
    and use them to label the Pulse shapes data. The labelled files will be stored
    in the data_destination 
    listmode_source (str): The path to the manually processed Listmodes (there is 
    one file per acquisition and per functional group)
    pulse_source (str) : The path to Default Pulse shapes data
    flr_num (int): Either 6 or 25. Indicate whether the curves have to be generated from FLR6 or FLR25 files
    is_untreated (bool): If the remaining indices in the Pulse shapes are labelled noise particle (False) or just untreated particles (True)
    ---------------------------------------------------------------------------------------------
    returns (None): Write the labelled Pulse shapes on hard disk
    '''
    assert (flr_num == 5) or (flr_num == 30) # 30 for OSCAHR ?

    log_file = data_destination + "/pred_logs.txt" # Register where write the already predicted files
    if not(os.path.isfile(log_file)):
        open(data_destination + '/pred_logs.txt', 'w+').close()

    # Defining the regex 
    date_regex = "(?:flr|FLR)" + str(flr_num) + " (20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:u|h)[0-9]{2})_[A-Za-z ()]+"

    listmode_files = [f for f in os.listdir(listmode_source)]
    listmode_files = [f for f in listmode_files if re.search("^listmode",f) and re.search("csv",f)]
    listmode_files = [f for f in listmode_files if re.search("(?:flr|FLR)" + str(flr_num), f)]

    dates = set([re.search(date_regex, f).group(1) for f in listmode_files])

    pulse_regex = "_([a-zA-Z0-9 ()]+)_Pulses.csv"  
    pulse_files = [f for f in os.listdir(pulse_source)]
    pulse_files = [f for f in pulse_files if re.search(pulse_regex,f) and re.search("csv",f)]
    pulse_files = [f for f in pulse_files if re.search("(?:flr|FLR)" + str(flr_num),f)]

    for date in dates: # For each sampling phase
        print("Processing:", date)

        ### Check if the acquisition has already been formatted
        with open(log_file, "r") as file:
            if date + '_FLR' + str(flr_num) in file.read(): 
                print("Already formatted")
                continue
        
        #### If the file has not already been formatted
        listmode_label = pd.DataFrame()
        clus_name_regex = '_((?:[a-zA-Z0-9 ()_]+){1,3})\.csv'

        # Get the info about each particule and its cluster label
        date_datasets_titles = [t for t in listmode_files if re.search(date, t)]
         
        # For each listmode file, i.e. for each functional group
        for title in date_datasets_titles:
            print(title)
            
            try:
                df = pd.read_csv(listmode_source + '/' + title, sep = ';', dtype = np.int64, usecols = [0])
            except ValueError: # If the data are in European format ("," stands for decimals and not thousands)
                df = pd.read_csv(listmode_source + '/' + title, sep = ';', dtype = np.int64, thousands='.', decimal=',', usecols = [0])
            
            # No formatting zeros

            # Add the date of the extraction
            df["date"] = date
            
            # Get the name of the cluster from the file name
            clus_name = re.search(clus_name_regex, title).group(1) 
            df["cluster"] = clus_name
            
            #df.set_index("ID", inplace = True)

            listmode_label = listmode_label.append(df)
        listmode_label.columns = ['Particle ID', 'date', 'cluster']
        
        # Getting the Pulse data from the corresponding Pulse shape files
        pulse_file = [f for f in pulse_files if re.search(date, f)][0] # Dirty
        
        print("Loading Pulse data")
        df = pd.read_csv(pulse_source + '/' + pulse_file, sep = ';', decimal = ',', engine = 'python') 

        df["date"] = date
        formating_lines = (df['Particle ID'] == 0) & (df['FWS'] == 0)
        
        df = df[~formating_lines] # Deleting formatting zeros
        
        existing_indices = listmode_label.index
        df.set_index("Particle ID", inplace = True)
        
        if is_untreated: # Extract only the data of the particles existing in the Listmodes
            df = df.loc[existing_indices] # Keep only the treated particles
            listmode_label = listmode_label.replace('Unidentified', 'noise') # Treat undefined as noise 
            listmode_label = df.reset_index().merge(listmode_label, on = ['Particle ID', 'date'])
            
        else: # The remaining particles are explictly treated as noise
            raise RuntimeError('is_untreated == False Not implemented')
            #noise_indices = list(set(df.index) - set(existing_indices)) # Determining the noise particles indices
            #df = df.loc[noise_indices] 
            #clus_name = "noise"
            #df["cluster"] = clus_name
            #listmode_label = listmode_label.append(df, sort = False) 
            
        listmode_label.to_csv(data_destination + '/Labelled_Pulse' + str(flr_num) + '_' + date + '.csv') # Store the data on hard disk
        
        # Mark that the file has been formatted
        with open(log_file, "a") as file:
            file.write(date + '_FLR' + str(flr_num) + '\n')
