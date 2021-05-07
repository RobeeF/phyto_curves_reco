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
    # Dirty assertion
    #assert (flr_num == 6) or (flr_num == 7) or (flr_num == 11) or (flr_num == 25) 
    
    nb_files_already_processed = 0
    log_file = data_destination + "/pred_logs.txt" # Register where write the already predicted files
    if not(os.path.isfile(log_file)):
        open(data_destination + '/pred_logs.txt', 'w+').close()
    else:
        with open(data_destination + '/pred_logs.txt', 'r') as file: 
            nb_files_already_processed = len(file.readlines())

    
    files_title = [f for f in os.listdir(data_source)]
    # Keep only the interesting csv files
    #flr_title = [f for f in files_title if re.search("FLR" + str(flr_num) + ' ',f) and re.search("csv",f) ]
    flr_title = [f for f in files_title if re.search("FLR" + str(flr_num),f) and re.search("csv",f) ]

    pulse_titles_clus = [f for f in flr_title if  re.search("Pulse",f) and not(re.search("Default",f))]
    pulse_titles_default = [f for f in flr_title if  re.search("Pulse",f) and re.search("Default",f)]

    # Defining the regex
    #date_regex = "FLR" + str(flr_num) + " (20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}h[0-9]{2})_[A-Za-z ()]+"

    info_regex = "FLR" + str(flr_num) + "(?:IIF)*\s*(20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:h|u)[0-9]{2})_([a-zA-Z0-9µ ()_-]+)_[Pp]ulses.csv"
    #date_regex = "FLR" + str(flr_num) + "(?:IIF)*\s*(20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:h|u)[0-9]{2})_[A-Za-z ()]+"
    #pulse_regex = "_([a-zA-Z0-9µ ()_-]+)_Pulses.csv"  

    dates = set([re.search(info_regex, f).group(1) for f in flr_title if  re.search("Pulse",f)])
    
    cluster_classes = list(set([re.search(info_regex, cc).group(2) for cc in pulse_titles_clus]))


    if len(pulse_titles_default) != 0:
        cluster_classes += ['Unassigned Particles']
     
    nb_acquisitions = len(dates)
    
    if nb_acquisitions == 0:
        print('No file found...')
    
    #date = list(dates)[0]
    for date in dates: # For each acquisition
        print(nb_files_already_processed + 1, '/', nb_acquisitions, "files have already been processed")
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
            clus_name = re.search(info_regex, title).group(2) 
            print(clus_name)
            
            try:
                df = pd.read_csv(data_source + '/' + title, sep = ';', dtype = np.float64)
            except ValueError: # If the data are in European format ("," stands for decimals and not thousands)
                try:
                    df = pd.read_csv(data_source + '/' + title, sep = ';', dtype = np.float64, thousands='.', decimal=',')
                except ValueError:
                    df = pd.read_csv(data_source + '/' + title, sep = ',', dtype = np.float64)
                except pd.errors.EmptyDataError:
                    print('Empty dataset')
                    continue

            df = df[df.values.sum(axis=1) != 0] # Delete formatting zeros
                                            
            # Add the date of the extraction
            df["date"] = date
            
            # Add the cluster name 
            df["cluster"] = clus_name
            
            if 'ID' in df.columns:# Handle cytoclus3 formatting
                df = df.rename(columns={'ID': 'Particle ID'})
                
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
                except ValueError:
                    df = pd.read_csv(data_source + '/' + title, sep = ',', dtype = np.float64)
                except pd.errors.EmptyDataError:
                    print('Empty dataset')
                    continue
            
            df = df[df.values.sum(axis=1) != 0] # Delete formatting zeros
            df["date"] = date
            
            existing_indices = pulse_data[pulse_data['date']==date].index
    
            if 'ID' in df.columns: # Handle cytoclus3 formatting
                df = df.rename(columns={'ID': 'Particle ID'})
                
            df.set_index("Particle ID", inplace = True)
            noise_indices = list(set(df.index) - set(existing_indices)) # Determining the noise particles indices
            df = df.loc[noise_indices] # Keep only the noise particles 
        
            clus_name = 'Unassigned Particles'
            df["cluster"] = clus_name
        
            pulse_data = pulse_data.append(df)
        else:
            raise RuntimeError('No Default file to deduce the noise particles from')

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
    assert (flr_num == 6) or (flr_num == 11) or (flr_num == 25) 
    
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
