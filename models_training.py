# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:32:19 2019

@author: Utilisateur
"""

import numpy as np
import os 
import pandas as pd

from time import time
from collections import Counter

from scipy.integrate import trapz
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, precision_score


os.chdir('C:/Users/rfuchs/Documents/GitHub/planktonPipeline')
cluster_classes = pd.read_csv('nomenclature.csv')['Nomenclature'].tolist()

from models import model13

os.chdir('C:/Users/rfuchs/Documents/GitHub/planktonPipeline/extract_Pulse_values')

from pred_functions import predict
from losses import categorical_focal_loss, CB_loss
from viz_functions import plot_2D, plot_2Dcyto


raw_data_source = 'L1_FUMSECK'
cleaned_data_source = 'L2_FUMSECK'

os.chdir('C:/Users/rfuchs/Documents/cyto_classif')

##############################################################################################
######################### Train Model 13 on FUMSECK Data ####################################
##############################################################################################

from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours

tn = pd.read_csv('train_test_nomenclature_FUMSECK.csv')
tn.columns = ['Particle_class', 'label']


# Load data
X_train = np.load('FUMSECK_L3/X_train610.npy')
y_train = np.load('FUMSECK_L3/y_train610.npy')

X_valid = np.load('FUMSECK_L3/X_valid610.npy')
y_valid = np.load('FUMSECK_L3/y_valid610.npy')

X_test = np.load('FUMSECK_L3/X_test610.npy')
y_test = np.load('FUMSECK_L3/y_test610.npy')

#========================================
# (Optional) ENN : delete dirty examples
#========================================

X_integrated = trapz(X_train, axis = 1)
X_integrated = pd.DataFrame(X_integrated, columns = ['SWS','FWS', 'FL Orange', 'FL Red', 'Curvature'])
y = y_train.argmax(1)
  
# ENN for cleaning data
enn = EditedNearestNeighbours()
X_rs, y_rs = enn.fit_resample(X_integrated, y) 

X_train = X_train.take(enn.sample_indices_, axis = 0)
y_train = y_train.take(enn.sample_indices_, axis = 0)

#plot_2Dcyto(X_rs, y_rs, tn, 'FWS', 'FL Red')
#plot_2Dcyto(X_integrated, y, tn, 'FWS', 'FL Red')

#========================================================
# RUS: Delete random observations from majority classes
#========================================================

balancing_dict = Counter(np.argmax(y_train,axis = 1))
for class_, obs_nb in balancing_dict.items():
    if obs_nb > 3000:
        balancing_dict[class_] = 3000


rus = RandomUnderSampler(sampling_strategy = balancing_dict)
ids = np.arange(len(X_train)).reshape((-1, 1))
ids_rs, y_train = rus.fit_sample(ids, y_train)
X_train = X_train[ids_rs.flatten()] 


w = 1/ np.sum(y_valid, axis = 0)
w = np.where(w == np.inf, np.max(w[np.isfinite(w)]) * 2 , w)
w = w / w.sum() 


batch_size = 128
STEP_SIZE_TRAIN = (len(X_train) // batch_size) + 1 
STEP_SIZE_VALID = (len(X_valid) // batch_size) + 1 

cffnn = model13(X_train, y_train, dp = 0.2)
ENN_check = ModelCheckpoint(filepath='tmp/weights_ENN.hdf5', verbose = 1, save_best_only=True)

epoch_nb = 1

history = cffnn.fit(X_train, y_train, validation_data=(X_valid, y_valid), \
                    steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                    epochs = epoch_nb, callbacks = [ENN_check], class_weight = w, shuffle=True)

cffnn.load_weights('tmp/weights_ENN.hdf5')



#### Compute accuracies #####

# Compute train accuracy
preds = np.argmax(cffnn.predict(X_train), axis = 1)
true = np.argmax(y_train, axis = 1)
acc = np.mean(preds == true)
print('Accuracy on train data !', acc)
print('Macro accuracy is', precision_score(true, preds, average='weighted'))
print(confusion_matrix(true, preds, labels = list(range(len(tn)))))

# Compute valid accuracy
preds = np.argmax(cffnn.predict(X_valid), axis = 1)
true = np.argmax(y_valid, axis = 1)
acc = np.mean(preds == true)
print('Accuracy on valid data !', acc)
print('Weighted accuracy is', precision_score(true, preds, \
                        average='weighted', zero_division = 0))
print(confusion_matrix(true, preds, labels = list(range(len(tn)))))

# Compute test accuracy
preds = np.argmax(cffnn.predict(X_test), axis = 1)
true = np.argmax(y_test, axis = 1)

acc = np.mean(preds == true)
print('Accuracy on test data !', acc)
print('Macro accuracy is', precision_score(true, preds, average='macro'))
print(confusion_matrix(true, preds, labels = list(range(len(tn)))))

# Good model : Save model
cffnn.save('ENN_LottyNet_FUMSECK')

############################################################################################################
################## Fine tune the LottyNet_FUMSECK on Endoume first week data ###############################
############################################################################################################

os.chdir('C:/Users/rfuchs/Documents/GitHub/planktonPipeline/extract_Pulse_values')

from keras.models import load_model
import numpy as np
from keras import metrics

fumseck = load_model('trained_models/LottyNet_FUMSECK')

#=================================================
# Loading Endoume first week data (the dirty way)
#=================================================

os.chdir('C:/Users/rfuchs/Documents/cyto_classif')

source = "SSLAMM/Week1"

files = os.listdir(source)
train_files = files[2:]
valid_files = [files[0]]
test_files = [files[1]] # Last one one the 21 files

#===============================================================
# Former model prediction for comparison
#===============================================================

predict(source + '/' + test_files[0], source, fumseck, tn)

preds = pd.read_csv(source + '/original/Pulse6_2019-09-18 15h59.csv')

q1 = 'Total FWS'
q2 = 'Total FLR'

plot_2D(preds, tn, q1, q2)

#===============================================================
# Model preparation for fine-tuning
#===============================================================

# Freeze the first layers and retrain
for layer in fumseck.layers[:5]:
    layer.trainable = False


ad = adam(lr=1e-3)
fumseck.compile(optimizer=ad, loss='categorical_crossentropy', \
                metrics=[metrics.categorical_accuracy])
    
#================================================================
# Data importation and model fitting
#================================================================

    
X_train_SLAAMM = np.load('FUMSECK_L3/X_trainFLR6_SSLAMM.npy')
y_train_SLAAMM = np.load('FUMSECK_L3/y_trainFLR6_SSLAMM.npy')

X_valid_SLAAMM = np.load('FUMSECK_L3/X_validFLR6_SSLAMM.npy')
y_valid_SLAAMM = np.load('FUMSECK_L3/y_validFLR6_SSLAMM.npy')

X_test_SLAAMM = np.load('FUMSECK_L3/X_testFLR6_SSLAMM.npy')
y_test_SLAAMM = np.load('FUMSECK_L3/y_testFLR6_SSLAMM.npy')
 

# Keep the weights w unchanged  

batch_size = 128
STEP_SIZE_TRAIN = (len(X_train_SLAAMM) // batch_size) + 1 
STEP_SIZE_VALID = (len(X_valid_SLAAMM) // batch_size) + 1 

blcd_check = ModelCheckpoint(filepath='tmp/weights_fum_n_slaamm.hdf5', verbose = 1, save_best_only=True)

epoch_nb = 6
for i in range(epoch_nb):
    
    history = fumseck.fit(X_train_SLAAMM, y_train_SLAAMM, validation_data=(X_valid_SLAAMM, y_valid_SLAAMM), \
                        steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                        epochs = 1, callbacks = [blcd_check])


fumseck.load_weights("tmp/weights_fum_n_slaamm.hdf5")

#=================================
# Visualising the results
#==================================

predict(source + '/' + test_files[0], source, fumseck, tn)

preds = pd.read_csv(source + 're_train/Pulse6_2019-09-18 15h59.csv')

q1 = 'Total FWS'
q2 = 'Total FLR'

plot_2D(preds, tn, q1, q2)


##############################################################################################
######################### Train Model 13 on Endoume Data ####################################
##############################################################################################

from collections import Counter

cluster_classes = ['airbubble', 'cryptophyte', 'hsnano', 'microphytoplancton',
       'nanoeucaryote', 'picoeucaryote', 'prochlorococcus',
       'synechococcus', 'unassigned particle']

# Load data
X_train = np.load('SSLAMM_L3/X_train.npy')
y_train = np.load('SSLAMM_L3/y_train.npy')

X_valid = np.load('SSLAMM_L3/X_valid.npy')
y_valid = np.load('SSLAMM_L3/y_valid.npy')

X_test = np.load('SSLAMM_L3/X_test.npy')
y_test = np.load('SSLAMM_L3/y_test.npy')

tn = pd.read_csv('train_test_nomenclature_SSLAMM.csv')
tn.columns = ['Particle_class', 'label']


#======================================
# Small viz 
#======================================*
X = trapz(X_train, axis = 1)
X = pd.DataFrame(X, columns = ['SWS','FWS', 'FL Orange', 'FL Red', 'Curvature'])
y = y_train.argmax(1)
y = np.array([list(tn[tn['label'] == yi]['Particle_class'])[0] for yi in y])

q1 = 'FWS'
q2 = 'FL Red'

plot_2Dcyto(X, y, tn, q1, q2)

#========================================
# Fitting model
#========================================

w = 1/ np.sum(y_valid, axis = 0)
w = np.where(w == np.inf, np.max(w[np.isfinite(w)]) * 2 , w)
#w[-3] = w[-3] * 1.2
w = w / w.sum() 

batch_size = 64
STEP_SIZE_TRAIN = (len(X_train) // batch_size) + 1 
STEP_SIZE_VALID = (len(X_valid) // 128) + 1 

sslamm_clf = model13(X_train, y_train, dp = 0.2)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
ENN_check = ModelCheckpoint(filepath='tmp/weights_sslamm_sup.hdf5',\
                            verbose = 1, save_best_only=True)


ad = adam(lr=1e-2)
sslamm_clf.compile(optimizer=ad, loss='categorical_crossentropy', \
                metrics=['accuracy'])

epoch_nb = 10

history = sslamm_clf.fit(X_train, y_train, validation_data=(X_valid, y_valid), \
                    steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                    epochs = epoch_nb, callbacks = [ENN_check, es], class_weight = w,\
                        shuffle=True)

sslamm_clf.load_weights('tmp/weights_sslamm_sup.hdf5')



#### Compute accuracies #####

# Compute train accuracy
preds = np.argmax(sslamm_clf.predict(X_train), axis = 1)
true = np.argmax(y_train, axis = 1)
acc = np.mean(preds == true)
print('Accuracy on train data !', acc)
print('Weighted accuracy is', precision_score(true, preds, average='weighted'))
print(confusion_matrix(true, preds, labels = list(range(len(tn)))))

# Compute valid accuracy
preds = np.argmax(sslamm_clf.predict(X_valid), axis = 1)
true = np.argmax(y_valid, axis = 1)
acc = np.mean(preds == true)
print('Accuracy on valid data !', acc)
print('Weighted accuracy is', precision_score(true, preds, \
                        average='weighted', zero_division = 0))
print(confusion_matrix(true, preds, labels = list(range(len(tn)))))

# Compute test accuracy
preds = np.argmax(sslamm_clf.predict(X_test), axis = 1)
true = np.argmax(y_test, axis = 1)

acc = np.mean(preds == true)
print('Accuracy on test data !', acc)
print('Macro accuracy is', precision_score(true, preds, average='macro'))
print(confusion_matrix(true, preds, labels = list(range(len(tn)))))
    
# Good model : Save model
sslamm_clf.save('LottyNet_SSLAMM')


##############################################################################################
############################ Try with focal loss  ############################################
##############################################################################################

batch_size = 64
STEP_SIZE_TRAIN = (len(X_train) // batch_size) + 1 
STEP_SIZE_VALID = (len(X_valid) // 128) + 1 

sslamm_clf = model13(X_train, y_train, dp = 0.2)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
ENN_check = ModelCheckpoint(filepath='tmp/weights_sslamm_focal.hdf5',\
                            verbose = 1, save_best_only=True)


ad = adam(lr=1e-2)
sslamm_clf.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], \
                   metrics=["accuracy"], optimizer=ad)


epoch_nb = 10

history = sslamm_clf.fit(X_train, y_train, validation_data=(X_valid, y_valid), \
                    steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                    epochs = epoch_nb, callbacks = [ENN_check, es], class_weight = w,\
                        shuffle=True) # weights ?

sslamm_clf.load_weights('tmp/weights_sslamm_focal.hdf5')


# Compute test accuracy
start = time()
preds = np.argmax(sslamm_clf.predict(X_test), axis = 1)
end = time()
print(end - start)
true = np.argmax(y_test, axis = 1)

acc = np.mean(preds == true)
print('Accuracy on test data !', acc)
print('Macro accuracy is', precision_score(true, preds, average='macro'))
print(confusion_matrix(true, preds, labels = list(range(len(tn)))))
    

X = pd.DataFrame(trapz(X_test, axis = 1), \
                         columns = ['SWS','Total FWS', 'FL Orange', 'Total FLR', 'Curvature'])
y = y_test.argmax(1)
plot_2Dcyto(X,y , tn, 'Total FWS', 'Total FLR', colors = None)

sslamm_clf.save('LottyNet_SSLAMM_focal')


##############################################################################################
############################### Try focal loss on few files ##################################
##############################################################################################

os.chdir('C:/Users/rfuchs/Documents/cyto_classif')


cluster_classes = ['airbubble', 'cryptophyte', 'hsnano', 'microphytoplancton',
       'nanoeucaryote', 'picoeucaryote', 'prochlorococcus',
       'synechococcus', 'unassigned particle']


X_train = np.load('SSLAMM_L3/X_train_umbal_SLAMM')
y_train = np.load('SSLAMM_L3/y_train_umbal_SLAMM')

X_valid = np.load('SSLAMM_L3/X_valid_umbal_SLAMM')
y_valid = np.load('SSLAMM_L3/y_valid_umbal_SLAMM')

X_test = np.load('SSLAMM_L3/X_test_umbal_SLAMM')
y_test = np.load('SSLAMM_L3/y_test_umbal_SLAMM')
    
    
batch_size = 64
STEP_SIZE_TRAIN = (len(X_train) // batch_size) + 1 
STEP_SIZE_VALID = (len(X_valid) // 128) + 1 

sslamm_clf = model13(X_train, y_train, dp = 0.2)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
ENN_check = ModelCheckpoint(filepath='tmp/weights_sslamm_focal_onefile.hdf5',\
                            verbose = 1, save_best_only=True)


ad = adam(lr=1e-2)
sslamm_clf.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], \
                   metrics=["accuracy"], optimizer=ad)


epoch_nb = 10

history = sslamm_clf.fit(X_train, y_train, validation_data=(X_valid, y_valid), \
                    steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                    epochs = epoch_nb, callbacks = [ENN_check, es],
                        shuffle=True)
    
sslamm_clf.load_weights('tmp/weights_sslamm_focal_onefile.hdf5')

# Compute test accuracy
preds = np.argmax(sslamm_clf.predict(X_test), axis = 1)
true = np.argmax(y_test, axis = 1)

acc = np.mean(preds == true)
print('Accuracy on test data !', acc)
print('Macro accuracy is', precision_score(true, preds, average='macro'))
print(confusion_matrix(true, preds, labels = list(range(len(tn)))))
    

X = pd.DataFrame(trapz(X_test, axis = 1), \
                         columns = ['SWS','Total FWS', 'FL Orange', 'Total FLR', 'Curvature'])
y = y_test.argmax(1)
plot_2Dcyto(X, preds , tn, 'Total FWS', 'Total FLR', colors = None)

sslamm_clf.save('LottyNet_SSLAMM_focal_one_file')


##############################################################################################
############################## Try bal_focal_loss on few files ###############################
##############################################################################################
from losses import CB_loss

os.chdir('C:/Users/rfuchs/Documents/cyto_classif')

tn = pd.read_csv('train_test_nomenclature_SSLAMM.csv')
tn.columns = ['Particle_class', 'label']

X_train = np.load('SSLAMM_L3/X_train_umbal_SLAMM.npy')
y_train = np.load('SSLAMM_L3/y_train_umbal_SLAMM.npy')

X_valid = np.load('SSLAMM_L3/X_valid_umbal_SLAMM.npy')
y_valid = np.load('SSLAMM_L3/y_valid_umbal_SLAMM.npy')

X_test = np.load('SSLAMM_L3/X_test_umbal_SLAMM.npy')
y_test = np.load('SSLAMM_L3/y_test_umbal_SLAMM.npy')
   
    
batch_size = 64
STEP_SIZE_TRAIN = (len(X_train) // batch_size) + 1 
STEP_SIZE_VALID = (len(X_valid) // 128) + 1 

sslamm_clf = model13(X_train, y_train, dp = 0.2)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
ENN_check = ModelCheckpoint(filepath='tmp/weights_sslamm_baloss_onefile.hdf5',\
                            verbose = 1, save_best_only=True)

sample_per_class = np.sum(y_valid, axis = 0) # Weights computed on valid, risk of overfit ?
sample_per_class = np.where(sample_per_class == 0, 1, sample_per_class)

ad = adam(lr=1e-2)
sslamm_clf.compile(loss=[CB_loss(sample_per_class)], \
                   metrics=["accuracy"], optimizer=ad)


epoch_nb = 10

history = sslamm_clf.fit(X_train, y_train, validation_data=(X_valid, y_valid), \
                    steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                    epochs = epoch_nb, callbacks = [ENN_check, es],
                        shuffle=True)
    
sslamm_clf.load_weights('tmp/weights_sslamm_baloss_onefile.hdf5')

# Compute test accuracy
preds = np.argmax(sslamm_clf.predict(X_test), axis = 1)
true = np.argmax(y_test, axis = 1)

acc = np.mean(preds == true)
print('Accuracy on test data !', acc)
print('Macro accuracy is', precision_score(true, preds, average='macro'))
print(confusion_matrix(true, preds, labels = list(range(len(tn)))))
print(precision_score(true, preds, average = None, labels = list(range(len(tn)))))   

X = pd.DataFrame(trapz(X_test, axis = 1), \
                         columns = ['SWS','Total FWS', 'FL Orange', 'Total FLR', 'Curvature'])
y = y_test.argmax(1)
y_train.argmax(0) + y_test.argmax(0) + y_valid.argmax(0)
plot_2Dcyto(X, preds , tn, 'Total FWS', 'Total FLR', colors = None)

sslamm_clf.save('LottyNet_SSLAMM_baloss_one_file')


##############################################################################################
############### Try bal_focal_loss on few (FUMSECK enriched) files ###########################
##############################################################################################

os.chdir('C:/Users/rfuchs/Documents/cyto_classif')

    
X_train = np.load('hybrid_L3/X_train.npy')
y_train = np.load('hybrid_L3/y_train.npy')

X_valid = np.load('hybrid_L3/X_valid.npy')
y_valid = np.load('hybrid_L3/y_valid.npy')  

X_test = np.load('hybrid_L3/X_test.npy')
y_test = np.load('hybrid_L3/y_test.npy')  # Il y a un soucis il devrait y avoir 0 

tn = pd.read_csv('train_test_nomenclature_SSLAMM.csv')
tn.columns = ['Particle_class', 'label']
    
batch_size = 128
STEP_SIZE_TRAIN = (len(X_train) // batch_size) + 1 
STEP_SIZE_VALID = (len(X_valid) // 128) + 1 

sslamm_clf = model13(X_train, y_train, dp = 0.2)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
ENN_check = ModelCheckpoint(filepath='tmp/weights_sslamm_baloss_enriched.hdf5',\
                            verbose = 1, save_best_only=True)

sample_per_class = np.sum(y_train, axis = 0) # Weights computed on valid, risk of overfit ?
sample_per_class = np.where(sample_per_class == 0, 1, sample_per_class)

ad = adam(lr=1e-2)
sslamm_clf.compile(loss=[CB_loss(sample_per_class)], \
                   metrics=["accuracy"], optimizer=ad)


epoch_nb = 4

history = sslamm_clf.fit(X_train, y_train, validation_data=(X_valid, y_valid), \
                    steps_per_epoch = STEP_SIZE_TRAIN, validation_steps = STEP_SIZE_VALID,\
                    epochs = epoch_nb, callbacks = [ENN_check, es],
                        shuffle=True)
    
sslamm_clf.load_weights('tmp/weights_sslamm_baloss_enriched.hdf5')

# Compute test accuracy
preds = np.argmax(sslamm_clf.predict(X_test), axis = 1)
true = np.argmax(y_test, axis = 1)


acc = np.mean(preds == true)
print('Accuracy on test data !', acc)
print('Macro accuracy is', precision_score(true, preds, average='macro'))
print(confusion_matrix(true, preds, labels = list(range(len(tn)))))
print(precision_score(true, preds, average = None, labels = list(range(len(tn)))))   

X = pd.DataFrame(trapz(X_test, axis = 1), \
                         columns = ['SWS','Total FWS', 'FL Orange', 'Total FLR', 'Curvature'])
y = y_test.argmax(1)

len(y)
plot_2Dcyto(X,  y, tn, 'Total FWS', 'Total FLR', colors = None)

Counter(y)

sslamm_clf.save('LottyNet_SSLAMM_baloss_enriched')


Counter(y_train.argmax(1))
Counter(preds)