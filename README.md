# phyto_curves_reco

`phyto_curves_reco` is a Python repository for automatic recognition of cytometric Phytoplankton Functional Groups (cPFG).
It enables to format the curves issued by a Cytosense (an Automated Flow Cytometer manufactured by Cytobuoy, b.v.) and predict the cPFG of each particle thanks to a Convolutional Neural Network.
The classes predicted correspond to six cPFGs described here\url{}:
- MICRO
- ORGNANO
- ORGPICOPRO
- REDNANO
- REDPICOEUK
- REDPICOPRO

In addition two other classes of particles can be identified by the Network
- Noise particles smaller than 1 μm.
- Noise particles bigger than 1 μm.

The final package will soon be available (February 2022).

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

# Extract the data
extract_non_labeled_curves(data_source, data_destination, flr_num = 6) # FLR6 acquisitions
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
For each acquisition, the file contains: The ID of each particle, the Total FLR, FLO/FLY, Curvature, FWS and SWS (Areas under the curve), the predicted class and the level of confidence of the prediction.

# Count the number of particles in each class

Then one can count the number of particles in each class. If you have only one acquisition no need to use the heavy loop presented in this section (just import the data with fastparquet and use the value_counts method on the resulting DataFrame).
If you have a whole time series instead, this loop could be useful.
