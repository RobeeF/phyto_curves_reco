# phyto_curves_reco

`phyto_curves_reco` is a Python library for automatic recognition of cytometric Phytoplankton Functional Groups (cPFG).
It enables to format the curves issued by a Cytosense (an Automated Flow Cytometer) and predict the cPFG of each particle thanks to a Convolutional Neural Network.
The classes predicted correspond to six cPFGs:
- Cryptophytes
- Microphytoplankton
- Nanoeukaryotes
- Picoeukaryotes
- Synechococcus
- Prochlorococcus

In addition three other classes of particles can be identified by the Network
- Noise particles smaller than 1 μm.
- Noise particles bigger than 1 μm.
- Airbubbles

The final package will soon be available (February 2021).

Before using the package, make sure you have CytoClus4 installed.
Then extract the Pulse shape files of the acquisitions you want to classify into a local folder.
In the following, this folder will be denoted by <source_folder>.

# Format the data and store them into fastparquet format
The first step is to format the data and store them into another local folder called thereafter <dest_folder>:

```python
import os
os.chdir('<the path where you have stored this package>/phyto_curves_reco')
from from_cytoclus_to_curves_values import extract_labeled_curves, extract_non_labeled_curves

data_source = <source_folder>
data_destination = <dest_folder>
flr_num = 25 # And 6

# Extract the data
extract_non_labeled_curves(data_source, data_destination, flr_num = 6) # trFLR6 acquisitions
extract_non_labeled_curves(data_source, data_destination, flr_num = 25) # FLR25 acquisitions
```

# Load the Convolutional Neural Network and predict the classes

Define a folder to store the predictions into (called <pred_folder> in the following).

```python
import re
import pandas as pd

from pred_functions import predict
from tensorflow.keras.models import load_model, model_from_json
from time import time

# Load the model in memory
cnn = load_model('<folder where the model is stored>/cnn_hyperopt_model_categ')

# Load the nomenclature of the classes
tn = pd.read_csv('train_test_nomenclature.csv')
tn.columns = ['Particle_class', 'label']

# Define where to look the data at and where to store preds
export_folder = '<dest_folder>'
export_files = os.listdir(export_folder)

pulse_regex = "_Pulse"
files_to_pred = [file for file in export_files if re.search(pulse_regex, file)] # The files containing the data to predict

# Create a log file in the destination folder: list of the already predicted files
preds_store_folder = "<pred_folder>"  # Where to store the predictions
log_path = preds_store_folder + "/pred_logs.txt" # Register where write the already predicted files

if not(os.path.isfile(log_path)):
    open(log_path, 'w+').close()

# Predict each file in turn
start_time = time()
nb_files_to_pred = len(files_to_pred)

for idx, file in enumerate(files_to_pred):
    print('Currently predicting ' + file + ' ' + str(idx))
    path = export_folder + '/' + file
    is_already_pred = False

    # Check if file has already been predicted
    with open(log_path, "r") as log_file:
        if file in log_file.read():
            is_already_pred = True

    if not(is_already_pred): # If not, perform the prediction
        # Predict the values
        predict(path, preds_store_folder,  cnn, tn, scale = False,\
            is_ground_truth = False)

        # Write in the logs that this file is already predicted
        with open(log_path, "a") as log_file:
            log_file.write(file + '\n')

        step_time = time()
        average_pred_time = (step_time - start_time) / (idx + 1)
        remaining_time = average_pred_time * (nb_files_to_pred - idx - 1)
        print('Average per file pred time', average_pred_time, idx, 'files already predicted')
        print('Remaining time before end of pred', remaining_time)

    else:
        print(file, 'already predicted')
```

The predictions are now available for each particle in the <pred_folder> folder.
For each acquisition, the file contains: The ID of each particle, the Total FLR, FLO, Curvature, FWS and SWS (Areas under the curve) and the predicted class.

# Count the number of particles in each class

Then one can count the number of particles in each class. If you have only one acquisition no need to use the heavy loop presented in this section (just import the data with pandas.read_csv and use the value_counts method on the resulting DataFrame).
If you have a whole time series instead, this loop could be useful.
To boost the prediction power, some post-processing rules can be used and are proposed in the following:

```python
# Fetch the files
pred_folder =  "<pred_folder>"
pred_files = os.listdir(pred_folder)

pulse_regex = "Pulse"
date_regex = "Pulse[0-9]{1,2}_(20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:u|h)[0-9]{2})"
flr_regex = 'Pulse([0-9]{1,2})'

files = [file for file in pred_files if re.search(pulse_regex, file)] # The files containing the data to predict

# The dataframe that store the results
phyto_ts = pd.DataFrame(columns = ['airbubble', 'cryptophyte', 'nanoeucaryote',\
                   'inf1microm_unidentified_particle', 'microphytoplancton',\
                'picoeucaryote', 'prochlorococcus', \
                'sup1microm_unidentified_particle', 'synechococcus', 'date', 'FLR'])

# Post processing rules:
SWS_noise_thr = 70
FWS_crypto_thr = 1E4
FWS_micros_thr = 2 * 10 ** 5

for file in files:
    path = pred_folder + '/' + file
    cl_count = pd.read_csv(path, usecols = ['Pred FFT Label', 'Total FWS', 'Total SWS'])

    # For cryptos, micros and phrochlo post processing
    real_cryptos = np.logical_and(cl_count['Total FWS'] >= FWS_crypto_thr, cl_count['Pred FFT Label'] == 'cryptophyte').sum()
    real_microphytos = np.logical_and(cl_count['Total FWS'] >= FWS_micros_thr, cl_count['Pred FFT Label'] == 'microphytoplancton').sum()
    false_microphytos = np.logical_and(cl_count['Total FWS'] < FWS_micros_thr, cl_count['Pred FFT Label'] == 'microphytoplancton').sum()
    false_noise = ((cl_count['Total FWS'] <= 100) & (cl_count['Total SWS'] >= SWS_noise_thr) & (cl_count['Pred FFT Label'] == 'inf1microm_unidentified_particle')).sum()

    cl_count = cl_count['Pred FFT Label'].value_counts()

    cl_count = pd.DataFrame(cl_count).transpose()
    flr_num = int(re.search(flr_regex, file).group(1))

    # Keep only "big" phyotplankton from FLR25 and "small" one from FLR6
    if flr_num == 25:
        for clus_name in ['synechococcus', 'prochlorococcus']:
            if clus_name in cl_count.columns:
                cl_count[clus_name] = 0

        # Post processing rules
        cl_count['cryptophyte'] = real_cryptos
        cl_count['microphytoplancton'] = real_microphytos
        cl_count['nanoeucaryote'] += false_microphytos

    elif flr_num == 6:
        for clus_name in ['picoeucaryote', 'cryptophyte', 'nanoeucaryote', 'microphytoplancton']:
            if clus_name in cl_count.columns:
                cl_count[clus_name] = 0

        # Post processing rules
        try:
          cl_count['prochlorococcus'] += false_noise
        except KeyError:
          cl_count['prochlorococcus'] = false_noise

    else:
        raise RuntimeError('Unknown flr number', flr_num)

    cl_count['inf1microm_unidentified_particle'] -= false_noise

    # Extract the date
    date = re.search(date_regex, file).group(1)

    # The timestamp is rounded to the closest 20 minutes    
    date = pd.to_datetime(date, format='%Y-%m-%d %Hh%M', errors='ignore')
    mins = date.minute

    if (mins >= 00) & (mins <= 30):
        date = date.replace(minute=00)


    elif (mins >= 31): # On arrondit à l'heure d'après
        if date.hour != 23:
            date = date.replace(hour= date.hour + 1, minute=00)
        else:
            try:
                date = date.replace(day = date.day + 1, hour = 00, minute=00)
            except:
                date = date.replace(month = date.month + 1, day = 1, hour = 00, minute=00)
    else:
      raise RuntimeError(date,'non handled')

    cl_count['date'] = date
    cl_count['FLR'] = flr_num

    phyto_ts = phyto_ts.append(cl_count)

# For those which have not both a FLR6 and a FLR25 file, replace the missing values by NaN
idx_pbs = pd.DataFrame(phyto_ts.groupby('date').size())
idx_pbs = idx_pbs[idx_pbs[0] == 1].index

for idx in idx_pbs:
    phyto_rpz_ts[phyto_rpz_ts['date'] == idx] = phyto_rpz_ts[phyto_rpz_ts['date'] == idx].replace(0, np.nan)

# Sum the particles counted in the FLR6 files with the associated FLR25 files
phyto_rpz_ts = phyto_ts.groupby('date').sum()
phyto_rpz_ts = phyto_rpz_ts.reset_index()
```

The DataFrame phyto_rpz_ts now contains all the predictions made for different acquisitions.
