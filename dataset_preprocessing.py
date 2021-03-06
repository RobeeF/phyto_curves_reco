import os
import numpy as np
import re
import pandas as pd
import fastparquet as fp

from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy


def gen_train_test_valid(source, cluster_classes, nb_files_tvt = [5, 4, 1],\
                         train_umbal_margin = 100, return_files = False, seed = None,\
                             flrs = [6,25]):
    ''' Generate a train balanced dataset and a test set with all observations
    source (str): The location of extracted (and formatted) Pulse files on disk
    cluster_classes (list of str): The classes used in the prediction task
    nb_files_tvt (list of int): the number of files used to generate respectively the train/valid/test sets
    flrs (list of size 2): The lowest and highest FLR thresholds 
    seed (all type): The seed to use to generate random samples
    ------------------------------------------------------------------------------------
    '''

    assert (np.array(nb_files_tvt) != 0).all()
    # Set the seed for train/test sets splitting
    np.random.seed(seed)

    # Encode the names of functional groups into integers
    le = LabelEncoder()
    le.fit(cluster_classes)

    # If validation set does not contain all classes or if the test set has not been generated 
    # by a FLR6 at least then redraw them
    not_all_classes_in_valid = True
    no_FLR6_in_valid = True
    no_FLR25_in_valid = True

    no_FLR6_in_test = True
    no_FLR25_in_test = True

    while not_all_classes_in_valid or no_FLR6_in_test or no_FLR25_in_test or no_FLR6_in_valid or no_FLR25_in_valid:
        # Fetching the formatted files
        files = os.listdir(source)
        files = [f for f in files if re.search("Labelled",f) and not(re.search('lock', f))]

        np.random.shuffle(files)

        # Assigning them between train, valid and test
        bounds = np.concatenate([[0], np.cumsum(nb_files_tvt)])

        train_files, valid_files, test_files = [files[bounds[i]:bounds[i + 1]] for i in range(3)]
        print('train', train_files)
        print('valid',valid_files)
        print('test', test_files)

        # Check that all classes are present in the validation set
        valid_classes = []
        for vfile in valid_files:
            pfile = fp.ParquetFile(source + '/' + vfile)
            vc = np.unique(pfile.to_pandas(columns=['cluster'])['cluster'])
            vc = homogeneous_cluster_names_swings(vc) # To decomment for SSLAMM
            valid_classes = valid_classes + vc # unlist vs if  homogeneous_cluster_names is uncommented

        valid_classes = np.unique(valid_classes)

        if (len(valid_classes) == len(cluster_classes)):
            not_all_classes_in_valid = False

        # Check that the valid files contains at least one FLR6 file which are more diversified than FLR25 files
        for vfile in valid_files:
            if re.search('Pulse' + str(flrs[0]), vfile):
                no_FLR6_in_valid = False
            if re.search('Pulse' + str(flrs[1]), vfile):
                no_FLR25_in_valid = False

        # Check that the test files contains at least one FLR6 file which are more diversified than FLR25 files
        for tfile in test_files:
            if re.search('Pulse' + str(flrs[0]), tfile):
                no_FLR6_in_test = False
            if re.search('Pulse' + str(flrs[1]), tfile):
                no_FLR25_in_test = False

    # Extract a balanced trained_dataset
    print('Generating train set')
    X_train, seq_len_list_train, y_train, pid_list_train, file_name_train, le_train = gen_dataset(source, \
                                cluster_classes, train_files, le, nb_obs_to_extract_per_group = train_umbal_margin, \
                                 to_balance = True, seed = None)


    # BE CAREFUL !!!! NO MORE UNDERSAMPLING HERE
    # Extract the valid dataset from full files
    print('Generating valid set')
    X_valid, seq_len_list_valid, y_valid, pid_list_valid, file_name_valid, le_valid = gen_dataset(source, \
                                cluster_classes, valid_files, le, nb_obs_to_extract_per_group = 10000000, \
                                to_balance = False, to_undersample = False, seed = None)
    
    # Extract the test dataset from full files
    print('Generating test set')
    X_test, seq_len_list_test, y_test, pid_list_test, file_name_test, le_test = gen_dataset(source, \
                                cluster_classes, test_files, le, nb_obs_to_extract_per_group = 10000000, \
                                to_balance = False, to_undersample = False, seed = None)


    # Write the nomenclature (Ex: Synnechochocus = 0, Prochlorrococus = 1 ...) used to generate the dataset
    pd.DataFrame(zip(le.classes_,le.transform(le.classes_)),\
                 columns = ['cluster', 'labels']).to_csv(source + '/train_test_nomenclature.csv',\
                 index = False)

    # Write the names of the test_files for latter vizualisation
    log_file = source + "/test_files_name.txt" # Create the file if it does not exist
    if not(os.path.isfile(log_file)):
        open(log_file, 'w+').close()

    # Mark that the file has been formatted
    with open(log_file, "a") as file:
        for tfile in test_files:
            file.write(tfile + '\n')

    if return_files:
        return [X_train, y_train, X_valid, y_valid, X_test, y_test], \
            [file_name_train, file_name_valid, file_name_test],\
            [pid_list_train, pid_list_valid, pid_list_test]
    else:
        return X_train, y_train, X_valid, y_valid, X_test, y_test


def gen_dataset(source, cluster_classes, files = [], le = None, nb_obs_to_extract_per_group = 1E7, \
                       to_balance = True, to_undersample = False, scale = False, seed = None):
    ''' Generate a balanced dataset from the cleaned Pulse files
    source (str): The location of extracted (and formatted) Pulse files on disk
    cluster_classes (list of str): The classes used in the prediction task
    files (list of str): If None extract the observations from all files of source, if list of str extract only from the names specified
    le (LabelEncoder object): None if no label encoder is provided.
    nb_obs_to_extract_per_group: Number of cells to extract for each group in each file
    seed (all type): The seed to use to generate random samples
    ------------------------------------------------------------------------------
    return (4 arrays): The dataset (X, y) the particle ids (pid) and the encoder of the groups names
    '''

    CURVES_DEFAULT_LEN = 120 # Standard curves lengths expected by the Network

    if le == None:
        le = LabelEncoder()
        le.fit(cluster_classes)

    # Fetching the formatted files if not provided
    if len(files) == 0:
        files = os.listdir(source)
        files = [f for f in files if re.search("Labelled",f) and not(re.search('lock', f))]

    X = []
    y = []
    pid_list = []
    seq_len_list = []
    file_name = []

    # Get the records of how many observations per class have already been included in the dataset(ignored if to_balance = False)
    balancing_dict = dict(zip(cluster_classes, np.full(len(cluster_classes), nb_obs_to_extract_per_group)))

    for idx, file in enumerate(files):
        print('File:', idx + 1, '/', len(files), '(', file, ')')
        pfile = fp.ParquetFile(source + '/' + file)
        df = pfile.to_pandas()

        if not('Particle ID' in df.columns):
            df = df.reset_index()
        if len(df) == 0:
          continue

        X_file, seq_len_file, y_file, pid_list_file = data_preprocessing(df, balancing_dict, CURVES_DEFAULT_LEN,  \
            to_balance = to_balance, to_undersample = to_undersample, seed = seed)

        if pd.isna(X_file).any():
            print('There were NaNs so the file was not treated')
            continue
        if len(X_file) == 0:
            continue

        X.append(X_file)
        y.append(y_file)
        pid_list.append(pid_list_file)
        seq_len_list.append(seq_len_file)
        file_name.append(np.repeat(file, len(X_file)))
        print('Nb of obs',len(np.concatenate(y)))

        if to_balance:
            # Defining the groups to sample in priority in the next sample: Those which have less observations
            balancing_dict = pd.Series(np.concatenate(y)).value_counts().to_dict()
            balancing_dict = {k: balancing_dict[k] if k in balancing_dict else 0 for k in cluster_classes}
            balancing_dict = {k: max(balancing_dict.values()) - balancing_dict[k] for k in cluster_classes}

    # Give the final form to the dataset
    X = np.vstack(X)

    if scale:
       X = scaler(X)

    y = np.concatenate(y)

    # Encode y and taking care of missing classes
    y = le.transform(y)
    y =  to_categorical(y, num_classes = len(cluster_classes))

    pid_list = np.concatenate(pid_list)
    seq_len_list = np.concatenate(seq_len_list)
    file_name = np.concatenate(file_name)

    # Sanity checks:
    assert y.shape[1] == len(cluster_classes)
    assert len(X) == len(y) == len(pid_list) == len(seq_len_list) == len(file_name)

    return X, seq_len_list, y, pid_list, file_name, le



def interp_sequences(sequences, max_len):
    ''' Interpolate sequences in order to reduce their length to max_len
        sequences (ndarray): The sequences to interpolate
        maxlen (int): The maximum length of the sequence: All sequences will be interpolated to match this length
        -------------------------------------------------------------------
        returns (ndarray): The interpolated sequences
    '''

    interp_obs = np.zeros((len(sequences), 5, 120))
    for idx, s in enumerate(sequences):
        original_len = s.shape[1]
        f = interp1d(np.arange(original_len), s, 'quadratic', axis = 1)
        interp_seq = f(np.linspace(0, original_len -1, num = max_len))
        interp_obs[idx] = interp_seq
    
    try:
        return np.stack(interp_obs)
    except ValueError:
        return np.array([])


def data_preprocessing(df, balancing_dict = None, max_len = None,  \
                           to_balance = True, to_undersample = False, seed = None):
    ''' Interpolates Pulse sequences and rebalance the dataset
    df (pandas DataFrame): The data container
    balancing_dict (dict): A dict that contains the desired quantities to extract for each group in order to obtain a balanced dataset. Only used if to_balance is True
    maxlen (int): The maximum length of the sequence: All sequences will be interpolated to match this length
    to_balance (Bool): Whether to balance or not the dataset
    to_undersample (Bool): Whether to undersample the data even if it is not to obtain a balanced data_set. Used only if to_balance == False
    seed (int): The seed to use to fix random results if wanted
    -----------------------------------------------------------------------------------------------------------------------------------
    returns (3 arrays): The dataset (X, y_list) y_list being unencoded labels and pid_list the list of corresponding particle ids
    '''

    # The final length of the sequences to fed the Neural Network
    DEFAULT_MAX_LEN = 120 # The default size of the
    DEFAULT_UDS_SIZE = 6000 # If decide to undersample, how many observations to keep in the end

    # Make the cluster names homogeneous and get the group of each particule
    df = homogeneous_cluster_names_swings(df) # To uncomment for SSLAMM
    pid_cluster = df[['Particle ID', 'cluster']].drop_duplicates()
    clus_value_count = pid_cluster.cluster.value_counts().to_dict()

    if to_balance:
        # Deleting non existing keys and adapting to the data in place
        balancing_dict  = {k: min(balancing_dict[k], clus_value_count[k]) for k in clus_value_count.keys()}

        # Undersampling to get a balanced dataset
        rus = RandomUnderSampler(random_state = seed, sampling_strategy = balancing_dict)

        pid_resampled, y_resampled = rus.fit_sample(pid_cluster['Particle ID'].values.reshape(-1,1), pid_cluster['cluster'])
        df_resampled = df.set_index('Particle ID').loc[pid_resampled.flatten()]

    else: # Keep the same distribution as in the original dataset
        if to_undersample: # Reduce the size of the dataset to speed things up
            pids = deepcopy(pid_cluster['Particle ID'].tolist())

            uds_size = np.min([DEFAULT_UDS_SIZE, len(pids)])
            pid_resampled = np.random.choice(pids, replace = False, size = uds_size)
            df_resampled = df.set_index('Particle ID').loc[pid_resampled.flatten()]

        else:
            df_resampled = df.set_index('Particle ID')

    # Reformatting the values
    obs_list = [] # The 5 curves
    pid_list = [] # The Particle ids
    y_list = [] # The class (e.g. 0 = prochlorocchoccus, 1= ...)
    seq_len_list = [] # The original length of the sequence


    for pid, obs in df_resampled.groupby('Particle ID'):
        # Sanity check: only one group for each particle
        try:
            assert(len(set(obs['cluster'])) == 1)
        except: # If a particle have been clustered in 2 groups them drop it
            print("Doublon de cluster", set(obs['cluster']), "pour l'obs ", pid)
            continue

        obs_list.append(obs.iloc[:,:5].values.T)
        seq_len_list.append(len(obs))
        pid_list.append(pid)
        y_list.append(list(set(obs['cluster']))[0])

    if max_len == None:
        max_len = DEFAULT_MAX_LEN


    obs_list = interp_sequences(obs_list, max_len)

    X = np.transpose(obs_list, (0, 2, 1))

    return X, seq_len_list, y_list, pid_list


def homogeneous_cluster_names(array):
    ''' Make homogeneous the names of the groups coming from the different Pulse files
    array (list, numpy 1D array or dataframe): The container in which the names have to be changed
    -----------------------------------------------------------------------------------------------
    returns (array): The array with the name changed and the original shape
    '''

    bubble_pat = '[_A-Za-z0-9?\-()]*bubble[_A-Za-z()0-9?\-]*'
    crypto_pat = '[_A-Za-z0-9?\-()]*crypto[_A-Za-z()0-9?\-]*'
    pico_pat = '[_A-Za-z0-9?\-()]*pico[_A-Za-z()0-9?\-]*'
    nano_pat = '[_A-Za-z0-9?\-()]*nano[_A-Za-z()0-9?\-]*'
    micro_pat = '[_A-Za-z0-9?\-()]*microp[_A-Za-z()0-9?\-]*'
    prochlo_pat = '[_A-Za-z0-9?\-()]*prochlo[_A-Za-z()0-9?\-]*'
    synecho_pat = '[_A-Za-z0-9?\-()]*synecho[_A-Za-z()0-9?\-]*'
    #noise_pat = '[_A-Za-z0-9?\-()]*noise[_A-Za-z()0-9?\-]*'
    lowfluo_pat = '[_A-Za-z0-9?\-()]*low[ _0-9]{0,1}fluo[_A-Za-z()0-9?\-]*'
    noiseinf_pat = '[_A-Za-z0-9?\-()]*noise[ _0-9]{0,1}in[fg][_A-Za-z()0-9?\-]*'
    noisesup_pat = '[_A-Za-z0-9?\-()]*noise[ _0-9]{0,1}sup[_A-Za-z()0-9?\-]*'
    nophyto_pat = '[_A-Za-z0-9?\-()]*nophyto[_A-Za-z()0-9?\-]*'
    unassigned_pat = '[_A-Za-z0-9?\-()]*[Uu]nassigned[ _A-Za-z()0-9?\-]*'
    unidentified_pat = '^unidentified_particles*'

    # Add regex for undertermined
    # Add regex for unassigned

    if type(array) == pd.core.frame.DataFrame:

        array['cluster'] = array.cluster.str.replace('Cryptophyceae','cryptophyte')

        array['cluster'] = array.cluster.str.replace('coccolithophorideae like','nanoeucaryote')
        array['cluster'] = array.cluster.str.replace('[_A-Za-z0-9?\-()]+naneu[_A-Za-z()0-9?\-]+','nanoeucaryote')

        array['cluster'] = array.cluster.str.replace('PPE 1','picoeucaryote')
        array['cluster'] = array.cluster.str.replace('PPE 2','picoeucaryote')


        array['cluster'] = array.cluster.str.replace('New Set 4','inf1microm_unidentified_particle')
        array['cluster'] = array.cluster.str.replace('New Set 7','inf1microm_unidentified_particle')

        array['cluster'] = array.cluster.str.replace('A_undetermined','noise')
        array['cluster'] = array.cluster.str.replace('C_undetermined','noise')

        array['cluster'] = array.cluster.str.replace('inf1um_unidentified_particle','inf1microm_unidentified_particle')
        array['cluster'] = array.cluster.str.replace('sup1um_unidentified_particle','sup1microm_unidentified_particle')

        array['cluster'] = array.cluster.str.replace('[_A-Za-z0-9?\-()]+syncho[_A-Za-z()0-9?\-]+','synechococcus')

        array['cluster'] = array['cluster'].str.replace(bubble_pat, 'airbubble', regex = True, case = False)
        array['cluster'] = array['cluster'].str.replace(crypto_pat, 'cryptophyte', regex = True, case = False)
        array['cluster'] = array['cluster'].str.replace(pico_pat, 'picoeucaryote', regex = True, case = False)
        array['cluster'] = array['cluster'].str.replace(nano_pat, 'nanoeucaryote', regex = True, case = False)
        array['cluster'] = array['cluster'].str.replace(micro_pat, 'microphytoplancton', regex = True, case = False)
        array['cluster'] = array['cluster'].str.replace(prochlo_pat, 'prochlorococcus', regex = True, case = False)
        array['cluster'] = array['cluster'].str.replace(synecho_pat, 'synechococcus', regex = True, case = False)
        array['cluster'] = array['cluster'].str.replace(unidentified_pat, 'noise', regex = True, case = False)

        array['cluster'] = array['cluster'].str.replace(noiseinf_pat,\
                        'inf1microm_unidentified_particle', regex = True, case = False)

        array['cluster'] = array['cluster'].str.replace(noisesup_pat,\
                        'sup1microm_unidentified_particle', regex = True, case = False)

        array['cluster'] = array['cluster'].str.replace(nophyto_pat,\
                        'sup1microm_unidentified_particle', regex = True, case = False)

        array['cluster'] = array['cluster'].str.replace(lowfluo_pat,\
                        'sup1microm_unidentified_particle', regex = True, case = False)

        array['cluster'] = array['cluster'].str.replace(unassigned_pat,\
                                                    'noise', regex = True, case = False)
            
        array['cluster'] = array.cluster.str.replace('C_noise1','inf1microm_unidentified_particle')


        array['cluster'] = array.cluster.str.replace('µ','micro')
        array['cluster'] = array.cluster.str.replace('es$','e') # Put in the names in singular form
        array['cluster'] = array.cluster.str.replace(' ','') # Put in the names in singular form

        array['cluster'] = array.cluster.str.lower()


    else:
        array = [re.sub('Cryptophyceae','cryptophyte', string) for string in array]

        array = [re.sub('coccolithophorideae like','nanoeucaryote', string) for string in array]
        array = [re.sub('[_A-Za-z0-9?\-()]+naneu[_A-Za-z()0-9?\-]+','nanoeucaryote', string) for string in array]

        array = [re.sub('PPE 1','picoeucaryotes', string) for string in array]
        array = [re.sub('PPE 2','picoeucaryotes', string) for string in array]


        array = [re.sub('A_undetermined','noise', string) for string in array]
        array = [re.sub('C_undetermined','noise', string) for string in array]

        array = [re.sub('New Set 7','inf1microm_unidentified_particle', string) for string in array]
        array = [re.sub('New Set 4','inf1microm_unidentified_particle', string) for string in array]


        array = [re.sub('[_A-Za-z0-9?\-()]+syncho[_A-Za-z()0-9?\-]+','synechococcus', string) for string in array]

        array = [re.sub(bubble_pat,'airbubble', string) for string in array]
        array = [re.sub(crypto_pat,'cryptophyte', string) for string in array]
        array = [re.sub(pico_pat,'picoeucaryote', string) for string in array]
        array = [re.sub(nano_pat,'nanoeucaryote', string) for string in array]
        array = [re.sub(micro_pat,'microphytoplancton', string) for string in array]
        array = [re.sub(prochlo_pat,'prochlorococcus', string) for string in array]
        array = [re.sub(synecho_pat,'synechococcus', string) for string in array]
        array = [re.sub(unidentified_pat, 'noise', string) for string in array]


        array = [re.sub(noiseinf_pat,'inf1microm_unidentified_particle', string) for string in array]
        array = [re.sub(noisesup_pat,'sup1microm_unidentified_particle', string) for string in array]

        array = [re.sub(nophyto_pat,'sup1microm_unidentified_particle', string) for string in array]
        array = [re.sub(lowfluo_pat,'sup1microm_unidentified_particle', string) for string in array]

        array = [re.sub(unassigned_pat, 'noise', string) for string in array]
        array = [re.sub('C_noise1','inf1microm_unidentified_particle', string) for string in array]


        array = [re.sub('µ','micro', string) for string in array]
        array = [re.sub('es$','e', string) for string in array]
        array = [re.sub(' ','', string) for string in array]

        array = [string.lower() for string in array]
        array = list(array)

    return array


def homogeneous_cluster_names_swings(array):
    ''' Make homogeneous the names of the groups coming from the different Pulse files of the Swings campaign
    array (list, numpy 1D array or dataframe): The container in which the names have to be changed
    -----------------------------------------------------------------------------------------------
    returns (array): The array with the name changed and the original shape
    '''
    if type(array) == pd.core.frame.DataFrame:
        array['cluster'] = array['cluster'].str.replace('bruitdefond', 'Unassigned Particles')
        array['cluster'] = array['cluster'].str.replace('([A-Za-z]+)_[A-Za-z]+', r'\1', regex = True, case = False)
        
    else:
        array = [re.sub('bruitdefond', 'Unassigned Particles', string) for string in array]
        array = [re.sub(r'([A-Za-z]+)_[A-Za-z]+', r'\1', string) for string in array]
        
    return array    

def scaler(X):
    ''' Scale the data. For the moment only minmax scaling is implemented'''
    X_mms = []

    # load data
    # create scaler
    scaler = MinMaxScaler()
    # fit and transform in one step
    for obs in X:
        normalized = scaler.fit_transform(obs)
        X_mms.append(normalized)

    X_mms = np.stack(X_mms)
    return X_mms


def extract_features_from_nn(dataset, pre_model):
    ''' Extract and flatten the output of a Neural Network
    dataset ((nb_obs,curve_length, nb_curves) array): The interpolated and scaled data
    pre_model (Keras model): The model without his head
    ---------------------------------------------------------------------
    returns ((nb_obs,nb_features) array): The features extracted from the NN
    '''

    features = pre_model.predict(dataset, batch_size=32)
    features_flatten = features.reshape((features.shape[0], -1))
    return features_flatten
