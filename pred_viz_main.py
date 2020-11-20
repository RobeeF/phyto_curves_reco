# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:06:49 2019

@author: Utilisateur
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import re
from sklearn.metrics import confusion_matrix, precision_score

os.chdir('C:/Users/rfuchs/Documents/GitHub/phyto_curves_reco')
#from losses import CB_loss


###################################################################################################################
# Visualize the predictions made on FUMSECK
###################################################################################################################

# Load nomenclature
tn = pd.read_csv('train_test_nomenclature_FUMSECK.csv')
tn.columns = ['Particle_class', 'label']

from pred_functions import predict
from viz_functions import plot_2D

from keras.models import load_model


folder = 'C:/Users/rfuchs/Documents/cyto_classif'
file = 'SSLAMM/Week1/Labelled_Pulse6_2019-09-18 14h35.parq'


date_regex = "(Pulse[0-9]{1,2}_20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:u|h)[0-9]{2})"
pred_file = 'Pulse6_2019-05-06 10h09.csv'
os.chdir(folder)

# Load pre-trained model
LottyNet = load_model('C:/Users/rfuchs/Documents/cyto_classif/LottyNet_FUMSECK') 

# Making formated predictions 
source_path = folder + '/' + file
dest_folder = folder
predict(source_path, folder, LottyNet, tn)

# Getting those predictions
preds = pd.read_csv(folder + '/' + pred_file)

np.mean(preds['True FFT id'] == preds['Pred FFT id'])
print(confusion_matrix(preds['True FFT id'] , preds['Pred FFT id']))

colors = ['#96ceb4', '#ffeead', '#ffcc5c', '#ff6f69', '#588c7e', '#f2e394', '#f2ae72', '#d96459']

#####################
# 2D plots
#####################

plot_2D(preds, tn, 'Total FLO', 'Total FLR', loc = 'lower right') # FLO vs FLR
plot_2D(preds, tn, 'Total FWS', 'Total FLR', loc = 'upper left')
plot_2D(preds, tn, 'Total SWS', 'Total FLR', loc = 'upper left')
plot_2D(preds, tn, 'Total SWS', 'Total FWS', loc = 'upper left')


####################
# Confusion matrix
####################

lab_tab = tn.set_index('id')['Label'].to_dict()
cluster_classes = list(lab_tab.values())
true = np.array(preds['True FFT id'])
pred_values = np.array(preds['Pred FFT id'])

preds['Pred FFT id'].value_counts()

pred_values[np.isnan(true)]
    
    

cm = confusion_matrix(preds['True FFT Label'], preds['Pred FFT Label'], cluster_classes)
cm = cm/cm.sum(axis = 1, keepdims = True)
cm = np.where(np.isnan(cm), 0, cm)
print(cm) 

fig = plt.figure(figsize = (16,16)) 
ax = fig.add_subplot(111) 
cax = ax.matshow(cm) 
plt.title('Confusion matrix of LottyNet_Full on a FLR6 file') 
fig.colorbar(cax) 
ax.set_xticklabels([''] + labels) 
ax.set_yticklabels([''] + labels) 
plt.xlabel('Predicted') 
plt.ylabel('True') 
plt.show()



#############################################################################################
# Plot the training set : FUMSECK
#############################################################################################

from scipy.integrate import trapz
from viz_functions import plot_2Dcyto

X = np.load('C:/Users/rfuchs/Documents/cyto_classif/FUMSECK_L3/X_train610.npy')
y = np.load('C:/Users/rfuchs/Documents/cyto_classif/FUMSECK_L3/y_train610.npy')

X_trapz = trapz(X, axis = 1)
X_trapz = pd.DataFrame(X_trapz, columns = ['SWS','FWS', 'FL Orange', 'FL Red', 'Curvature'])

y_label = y.argmax(1)
#y_str_label = np.array([tn[tn['label'] == yi]['Particle_class'].values[0] for yi in y_label])


colors = ['#96ceb4', 'gold', 'black', 'green', 'grey', 'red', 'purple', 'blue']
# Uncomment to mask some of the categories
#colors = ['black', 'white', 'white', 'white', 'white', 'white', 'white', 'white']


q1 = 'FL Red'
q2 = 'FL Orange'

plot_2Dcyto(X_trapz, y_label, tn, q1, q2, colors)

#============================================
# Try with Undersampling
#============================================

from imblearn.under_sampling import NearMiss, EditedNearestNeighbours
from collections import Counter
  
# ENN for cleaning data
ratio_list = list(set(range(8)) - set([0,2,4]))
enn = EditedNearestNeighbours(sampling_strategy = ratio_list)
X_rs, y_rs = enn.fit_resample(X_trapz, y_label)

ratio_dict = {}
for i in range(8):
    ratio_dict[i] = 400  

nm1 = NearMiss(version = 1, sampling_strategy = ratio_dict)
X_rs, y_rs = nm1.fit_resample(X_rs, y_rs)


plot_2Dcyto(X_rs, y_rs, tn, q1, q2, colors)

Counter(y_rs)


# Near miss: Taille trop mais la Version 1 est la meilleure
# Tomek taille pas assez dans le gras
# ENN non plus mais a l'effet escompté ! (Meilleur que Tomek)
# OSS vire carrément les pico et les synnecho oklm
# NeighbourhoodCleaningRule


#=====================================================
# Outliers detection with kmeans
#=====================================================
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_trapz, y_label)
y_knn = knn.predict(X_trapz)


plot_2Dcyto(X_trapz, y_knn, tn, q1, q2, colors)

Counter(y_rs)
len(y_rs)
len(y_label)


#############################################################################################
# Undersampling a random acquisition
#############################################################################################

import fastparquet as fp
from dataset_preprocessing import homogeneous_cluster_names, interp_sequences

folder = 'C:/Users/rfuchs/Documents/cyto_classif/FUMSECK_L2_fp'

pfile = fp.ParquetFile(folder + '/' + 'Labelled_Pulse6_2019-05-05 06h09.parq')
true = pfile.to_pandas()

true = homogeneous_cluster_names(true)
true = true.set_index('Particle ID')
grouped_df = true[['SWS','FWS', 'FL Orange', 'FL Red', 'Curvature']].groupby('Particle ID')

obs_list = [obs.values.T for pid, obs in grouped_df]
obs_list = interp_sequences(obs_list, 120)
X_true = trapz(obs_list, axis = 2)
X_true = pd.DataFrame(X_true, columns = ['SWS','FWS', 'FL Orange', 'FL Red', 'Curvature'])

y_true = true.groupby('Particle ID')['cluster'].apply(np.unique)
y_true = np.stack(y_true)[:,0]
y_enc = np.array([list(tn[tn['Particle_class'] == yi]['label'])[0] for yi in y_true])

ratio_list = list(set(range(8)) - set([0,2,4]))
enn = EditedNearestNeighbours(sampling_strategy = ratio_list)
X_rs, y_rs = enn.fit_resample(X_true, y_enc)


plot_2Dcyto(X_rs, y_rs, tn, q1, q2, colors)


Counter(y_true)
1 - len(y_rs) / len(y_label)



#############################################################################################
# Plot the training set : Endoume  (FLR 6 only)
#############################################################################################

from scipy.integrate import trapz
from pred_functions import plot_2Dcyto

X = np.load('C:/Users/rfuchs/Documents/cyto_classif/FUMSECK_L3/X_trainFLR6_SSLAMM.npy')
y = np.load('C:/Users/rfuchs/Documents/cyto_classif/FUMSECK_L3/y_trainFLR6_SSLAMM.npy')

X_trapz = trapz(X, axis = 1)
X_trapz = pd.DataFrame(X_trapz, columns = ['SWS','FWS', 'FL Orange', 'FL Red', 'Curvature'])

y_label = y.argmax(1)

q1 = 'FWS'
q2 = 'FL Red'

plot_2Dcyto(X_trapz, y_label, tn, q1, q2)

# Test set
X = np.load('C:/Users/rfuchs/Documents/cyto_classif/FUMSECK_L3/X_testFLR6_SSLAMM.npy')
y = np.load('C:/Users/rfuchs/Documents/cyto_classif/FUMSECK_L3/y_testFLR6_SSLAMM.npy')

X_trapz = trapz(X, axis = 1)
X_trapz = pd.DataFrame(X_trapz, columns = ['SWS','FWS', 'FL Orange', 'FL Red', 'Curvature'])

y_label = y.argmax(1)

q1 = 'FWS'
q2 = 'FL Red'

plot_2Dcyto(X_trapz, y_label, tn, q1, q2)

###############################################################################
# Pick some Endoume predictions and plot true vs pred 
###############################################################################

import re
import fastparquet as fp
from dataset_prepocessing import homogeneous_cluster_names


true_folder = 'C:/Users/rfuchs/Documents/SSLAMM_P1/SSLAMM_True_L1'
pred_folder = 'C:/Users/rfuchs/Documents/SSLAMM_P1/SSLAMM_L2'
graph_folder = 'C:/Users/rfuchs/Documents/SSLAMM_P1/graphs_true_pred/1_23_03_20'

nb_plots = 10
true_files = [f for f in os.listdir(true_folder) if re.search('parq', f)]
picked_true_files = np.random.choice(true_files, nb_plots, replace = False)

date_regex = "(Pulse[0-9]{1,2}_20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:u|h)[0-9]{2})"
titles = [re.sub('Pulse', 'FLR', re.search(date_regex, f).group(0)) for f in picked_true_files]
picked_pred_files = [re.search(date_regex, f).group(0) + '.csv' for f in picked_true_files]

acc = np.zeros(nb_plots)

for i, file in enumerate(picked_true_files):
    pfile = fp.ParquetFile(true_folder + '/' + file)
    true = pfile.to_pandas(columns=['Particle ID','cluster'])
    true = true.reset_index().drop_duplicates()
    true = homogeneous_cluster_names(true)
    true.columns = ['Particle ID', 'True FFT Label']
    
    pred = pd.read_csv(pred_folder + '/' + picked_pred_files[i])
    
    if len(pred) != len(true):
        raise RuntimeError('Problem on', file)
    
    true_pred = pd.merge(true, pred, on = 'Particle ID')
    acc[i] = np.mean(true_pred['True FFT Label'] == true_pred['Pred FFT Label'])
    print(acc[i])
    plot_2D(true_pred, tn, 'Total FWS', 'Total FLR', loc = 'upper left', title = graph_folder + '/' + titles[i])


np.mean(acc)
q1 = 'Total FWS'
q2 = 'Total FLR'
#fp.to_pandas
#pd.read()
#true_pred = pd.merge([true, preds, on = 'Particle id')
#np.mean(true_pred['True FFT id'] == true_pred['Pred FFT id'])
#plot_2D(preds, tn, 'Total FWS', 'Total FLR', loc = 'upper left') # Add savefig ? 
plt.savefig('cocuou')


L1_pred_folder = 'C:/Users/rfuchs/Documents/SSLAMM_P1/SSLAMM_L1'
pd.read_csv(L1_pred_folder + "/" + file)

pfile2 = fp.ParquetFile(true_folder + '/' + file)
a = pfile2.to_pandas()
    
a.loc[88.0]
true.loc[88.0]
set(a.index) - set(true.index)
set(true.index) - set(a.index)

#############################################################################################
# Plot predicted time series : 1st pred
#############################################################################################

from dataset_preprocessing import homogeneous_cluster_names
os.chdir('C:/Users/rfuchs/Documents/preds')

ts = pd.read_csv('pred2/P1/09_to_12_2019.csv')
ts['date'] =  pd.to_datetime(ts['date'], format='%Y-%m-%d %H:%M:%S')
ts = ts.set_index('date')

cols_plot = ts.columns
axes = ts[cols_plot].plot(alpha=0.5, linestyle='-', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Count')
    

# Formatting True time series for P1
true_ts = pd.read_csv('C:/Users/rfuchs/Documents/09_to_12_2019_true.csv', sep = ';', engine = 'python')
true_ts = true_ts[['Date','count', 'set']]
true_ts['Date'] =  pd.to_datetime(true_ts['Date'], format='%d/%m/%Y %H:%M:%S')
true_ts.columns = ['Date','count', 'cluster']
true_ts = homogeneous_cluster_names(true_ts)
true_ts['cluster'] = true_ts['cluster'].replace('default (all)', 'noise')

true_ts = true_ts.set_index('Date')


for cluster_name in ts.columns:
    if cluster_name in set(true_ts['cluster']):
        # Picoeuk comparison: (HighFLR are neglected)
        true_ts_clus = pd.DataFrame(true_ts[true_ts['cluster'] == cluster_name]['count'])
        true_ts_clus.columns = ['true_count']
        pred_ts_clus = pd.DataFrame(ts[cluster_name])
        pred_ts_clus.columns = ['pred_count']
        
        true_ts_clus.index = true_ts_clus.index.floor('H')
        pred_ts_clus.index = pred_ts_clus.index.floor('H')
        
        all_clus = true_ts_clus.join(pred_ts_clus)
        
        all_clus.plot(alpha=0.5, figsize=(17, 9), marker='.', title = cluster_name)
        plt.savefig('C:/Users/rfuchs/Desktop/pred_P1/' + cluster_name + '.png')
    else:
        print(cluster_name, 'is not in true_ts pred')

ts['microphytoplancton'].plot()


#############################################################################################
# Plot predicted time series : 2nd pred
#############################################################################################
from dataset_preprocessing import homogeneous_cluster_names
os.chdir('C:/Users/rfuchs/Documents/preds')


#==================================================================
# Old prediction import
#==================================================================
old_ts = pd.read_csv('pred1/P1/09_to_12_2019.csv')
old_ts['date'] =  pd.to_datetime(old_ts['date'], format='%Y-%m-%d %H:%M:%S')
old_ts = old_ts.set_index('date')

cols_plot = old_ts.columns
axes = old_ts[cols_plot].plot(alpha=0.5, linestyle='-', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Count')

#==================================================================
# New prediction import
#==================================================================

ts = pd.read_csv('pred2/P1/09_to_12_2019.csv')
ts['date'] =  pd.to_datetime(ts['date'], format='%Y-%m-%d %H:%M:%S')
ts = ts.set_index('date')
ts.columns = ['picoeucaryote', 'synechococcus', 'nanoeucaryote', 'cryptophyte',
       'noise', 'airbubble', 'microphytoplancton',
       'prochlorococcus']

cols_plot = ts.columns
axes = ts[cols_plot].plot(alpha=0.5, linestyle='-', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Count')

#==================================================================
# New prediction import with proba
#==================================================================

ts_proba = pd.read_csv('pred2/P1/09_to_12_2019_proba.csv')
ts_proba['date'] =  pd.to_datetime(ts_proba['date'], format='%Y-%m-%d %H:%M:%S')
ts_proba = ts_proba.set_index('date')
ts_proba.columns = ['picoeucaryote', 'synechococcus', 'nanoeucaryote', 'cryptophyte',
       'noise', 'airbubble', 'microphytoplancton',
       'prochlorococcus']

cols_plot = ts_proba.columns
axes = ts_proba[cols_plot].plot(alpha=0.5, linestyle='-', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Count')    

ts_proba['microphytoplancton']
ts['microphytoplancton']

#==================================================================
# True series import
#==================================================================

true_ts = pd.read_csv('pred2/P1/09_to_12_2019_true.csv', sep = ';', engine = 'python')
true_ts = true_ts[['Date','count', 'set']]
true_ts['Date'] =  pd.to_datetime(true_ts['Date'], format='%d/%m/%Y %H:%M:%S')
true_ts.columns = ['Date','count', 'cluster']
true_ts = homogeneous_cluster_names(true_ts)
true_ts['cluster'] = true_ts['cluster'].replace('default (all)', 'noise')

true_ts = true_ts.set_index('Date')

true_ts['nanoeucaryote']
true_ts_clus = pd.DataFrame(true_ts[true_ts['cluster'] == 'nanoeucaryote']['count'])
true_ts_clus.mean()

for cluster_name in ts.columns: 
    pred_ts_clus = pd.DataFrame(ts[cluster_name])
    pred_ts_clus.columns = ['pred_count']
    pred_ts_clus.index = pred_ts_clus.index.floor('H')

    
    old_pred_ts_clus = pd.DataFrame(old_ts[cluster_name])
    old_pred_ts_clus.columns = ['old_pred_count']
    old_pred_ts_clus.index = old_pred_ts_clus.index.floor('H')
    
    pred_ts_proba_clus = pd.DataFrame(ts_proba[cluster_name])
    pred_ts_proba_clus.columns = ['pred_count_proba']
    pred_ts_proba_clus.index = pred_ts_proba_clus.index.floor('H')

    
    if cluster_name in set(true_ts['cluster']):
        # Picoeuk comparison: (HighFLR are neglected)
        true_ts_clus = pd.DataFrame(true_ts[true_ts['cluster'] == cluster_name]['count'])
        true_ts_clus.columns = ['true_count']
           
        true_ts_clus.index = true_ts_clus.index.floor('H')
            
        all_clus = true_ts_clus.join(pred_ts_clus)
        all_clus = all_clus.join(old_pred_ts_clus)
        all_clus = all_clus.join(pred_ts_proba_clus)
        
        all_clus.plot(alpha=0.5, figsize=(17, 9), marker='.', title = cluster_name)
        plt.savefig('C:/Users/rfuchs/Desktop/pred_P1/' + cluster_name + '.png')
    else:
        all_clus = old_pred_ts_clus.join(pred_ts_clus)
        all_clus = all_clus.join(pred_ts_proba_clus)
        print(all_clus)
        all_clus.plot(alpha=0.5, figsize=(17, 9), marker='.', title = cluster_name)
        plt.savefig('C:/Users/rfuchs/Desktop/pred_P1/' + cluster_name + '.png')        
        print(cluster_name, 'is not in true_ts pred')


#############################################################################################
# Plot predicted time series : 3rd pred
#############################################################################################

from dataset_preprocessing import homogeneous_cluster_names
os.chdir('C:/Users/rfuchs/Documents/preds')


#==================================================================
# Old prediction import
#==================================================================
old_ts = pd.read_csv('pred2/P1/09_to_12_2019.csv')
old_ts['date'] =  pd.to_datetime(old_ts['date'], format='%Y-%m-%d %H:%M:%S')
old_ts = old_ts.set_index('date')
old_ts.columns = ['picoeucaryote', 'synechococcus', 'nanoeucaryote', 'cryptophyte',
       'noise', 'airbubble', 'microphytoplancton',
       'prochlorococcus']

cols_plot = old_ts.columns
axes = old_ts[cols_plot].plot(alpha=0.5, linestyle='-', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Count')

#==================================================================
# New prediction import
#==================================================================

ts = pd.read_csv('pred3/P1/09_to_12_2019.csv')
ts['date'] =  pd.to_datetime(ts['date'], format='%Y-%m-%d %H:%M:%S')
ts = ts.set_index('date')
ts.columns = ['picoeucaryote', 'synechococcus', 'nanoeucaryote', 'cryptophyte',
       'noise', 'airbubble', 'microphytoplancton',
       'prochlorococcus']

cols_plot = ts.columns
axes = ts[cols_plot].plot(alpha=0.5, linestyle='-', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Count')


#==================================================================
# True series import
#==================================================================

true_ts = pd.read_csv('pred2/P1/09_to_12_2019_true.csv', sep = ';', engine = 'python')
true_ts = true_ts[['Date','count', 'set']]
true_ts['Date'] =  pd.to_datetime(true_ts['Date'], format='%d/%m/%Y %H:%M:%S')
true_ts.columns = ['Date','count', 'cluster']
true_ts = homogeneous_cluster_names(true_ts)
true_ts['cluster'] = true_ts['cluster'].replace('default (all)', 'noise')

true_ts = true_ts.set_index('Date')


for cluster_name in ts.columns: 
    pred_ts_clus = pd.DataFrame(ts[cluster_name])
    pred_ts_clus.columns = ['pred_count']
    pred_ts_clus.index = pred_ts_clus.index.floor('H')

    
    #old_pred_ts_clus = pd.DataFrame(old_ts[cluster_name])
    #old_pred_ts_clus.columns = ['old_pred_count']
    #old_pred_ts_clus.index = old_pred_ts_clus.index.floor('H')
    
    #pred_ts_proba_clus = pd.DataFrame(ts_proba[cluster_name])
    #pred_ts_proba_clus.columns = ['pred_count_proba']
    #pred_ts_proba_clus.index = pred_ts_proba_clus.index.floor('H')

    
    if cluster_name in set(true_ts['cluster']):
        # Picoeuk comparison: (HighFLR are neglected)
        true_ts_clus = pd.DataFrame(true_ts[true_ts['cluster'] == cluster_name]['count'])
        true_ts_clus.columns = ['true_count']
           
        true_ts_clus.index = true_ts_clus.index.floor('H')
            
        all_clus = true_ts_clus.join(pred_ts_clus)
        #all_clus = all_clus.join(old_pred_ts_clus)
        #all_clus = all_clus.join(pred_ts_proba_clus)
        
        all_clus.plot(alpha=0.5, figsize=(17, 9), marker='.', title = cluster_name)
        #plt.savefig('C:/Users/rfuchs/Desktop/pred_P1/' + cluster_name + '.png')
    else:
        #all_clus = old_pred_ts_clus.join(pred_ts_clus)
        all_clus = pred_ts_clus
        #all_clus = all_clus.join(pred_ts_proba_clus)
        all_clus.plot(alpha=0.5, figsize=(17, 9), marker='.', title = cluster_name)
        #plt.savefig('C:/Users/rfuchs/Desktop/pred_P1/' + cluster_name + '.png')        
        print(cluster_name, 'is not in true_ts pred')
        
#############################################################################################
# Plot predicted time series : 4th pred
#############################################################################################

from dataset_preprocessing import homogeneous_cluster_names
os.chdir('C:/Users/rfuchs/Documents/preds')

#==================================================================
# Old prediction import
#==================================================================
old_ts = pd.read_csv('pred2/P1/09_to_12_2019.csv')
old_ts['date'] =  pd.to_datetime(old_ts['date'], format='%Y-%m-%d %H:%M:%S')
old_ts = old_ts.set_index('date')
old_ts.columns = ['picoeucaryote', 'synechococcus', 'nanoeucaryote', 'cryptophyte',
       'noise', 'airbubble', 'microphytoplancton',
       'prochlorococcus']

#==================================================================
# New prediction import
#==================================================================

#ts = pd.read_csv('pred4/P1/09_to_12_2019_inf_sup.csv')
#ts = pd.read_csv('pred4/P1/09_to_12_2019.csv')
ts = pd.read_csv('pred4/P1/09_to_12_2019_nouvelle_moulinette.csv')


ts['date'] =  pd.to_datetime(ts['date'], format='%Y-%m-%d %H:%M:%S')
ts = ts.set_index('date')

#==================================================================
# True series import
#==================================================================

true_ts = pd.read_csv('pred4/P1/09_to_12_2019_true_allFLR_Default.csv', sep = ';', engine = 'python')
#true_ts = pd.read_csv('pred4/P1/09_to_12_2019_true_allFLR.csv', engine = 'python', sep = ';')
#true_ts = pd.read_csv('pred4/P1/09_to_12_2019_true.csv', engine = 'python', sep = ';')


true_ts['Date'] =  pd.to_datetime(true_ts['tSBE'], format='%d/%m/%Y %H:%M') # tSBE ...
#true_ts['Date'] =  pd.to_datetime(true_ts['Datearrondi'], format='%d/%m/%Y %H:%M') # tSBE ...
#true_ts['Date'] =  pd.to_datetime(true_ts['tSBE'], format='%Y-%m-%d %H:%M:%S')

true_ts = true_ts[['Date','count', 'set']]
true_ts.columns = ['Date','count', 'cluster']

# Drop the NAs
true_ts = true_ts[~true_ts['cluster'].isna()]


# Delete the FLR25 nano and pico

true_ts = true_ts[true_ts['cluster'] != 'nanoeukFLR6']
true_ts = true_ts[true_ts['cluster'] != 'picoeukFLR6']
true_ts = true_ts[true_ts['cluster'] != 'picohighFLR6']
true_ts = true_ts[true_ts['cluster'] != 'cryptophytesFLR6']


true_ts['cluster'] = true_ts.cluster.str.replace('inf1Âµm_unidentified_particles','inf1microm_unidentified_particle')
true_ts['cluster'] = true_ts.cluster.str.replace('sup1Âµm_unidentified_particles','sup1microm_unidentified_particle')

true_ts = homogeneous_cluster_names(true_ts)

# Compute the count of each PFT once the names have been homogeneized
true_ts = true_ts.groupby(['Date', 'cluster']).sum().reset_index()

# Compute the number of noise particles
non_noise = true_ts[true_ts['cluster'] != 'defaultflr6'].groupby('Date').sum().reset_index()#['count']
default_count = true_ts[true_ts['cluster'] == 'defaultflr6']#.reset_index(drop = True)


#Try 1

noise = non_noise.merge(default_count, on = 'Date')
noise['count'] = noise['count_y'] - noise['count_x']
noise['Date'] = default_count['Date'].values
noise['cluster'] = 'noise'
noise = noise[['Date', 'cluster', 'count']]
true_ts = true_ts[true_ts['cluster'] != 'defaultflr6']
true_ts = true_ts.append(noise)


true_ts = true_ts.set_index('Date')
true_ts = true_ts.sort_index()

fontsize = 24

title_number = ['a) ', 'b) ', 'c) ', 'd) ', 'e) ', 'f) ']
interesting_classes = ['picoeucaryote', 'nanoeucaryote',
        'cryptophyte', 'microphytoplancton',
        'prochlorococcus', 'synechococcus']

for idx, cluster_name in enumerate(interesting_classes): 
    pred_ts_clus = pd.DataFrame(ts[cluster_name])
    pred_ts_clus.columns = ['pred_count']
    #pred_ts_clus.index = pred_ts_clus.index.floor('H')
    pred_ts_clus[pred_ts_clus['pred_count'] == 0] = np.nan
    pred_ts_clus = pred_ts_clus.fillna(method = 'ffill')

    #old_pred_ts_clus = pd.DataFrame(old_ts[cluster_name])
    #old_pred_ts_clus.columns = ['old_pred_count']
    #old_pred_ts_clus.index = old_pred_ts_clus.index.floor('H')

    # Translate the name into english
    printed_cluster_name = re.sub('plancton','plankton',cluster_name)
    printed_cluster_name = re.sub('eucar','eukar', printed_cluster_name)

    if cluster_name in set(true_ts['cluster']):
        # Picoeuk comparison: (HighFLR are neglected)
        true_ts_clus = pd.DataFrame(true_ts[true_ts['cluster'] == cluster_name]['count'])
        true_ts_clus.columns = ['true_count']
           
        true_ts_clus.index = true_ts_clus.index.floor('H')
            
        all_clus = true_ts_clus.join(pred_ts_clus)

    else:
        all_clus = pd.DataFrame(0, index = pred_ts_clus.index, columns = ['true_count'])
        all_clus = all_clus.join(pred_ts_clus)
        #all_clus = pred_ts_clus
               
        print(cluster_name, 'is not in true_ts pred')

    all_clus.columns = ['Manual Gating', 'Automatic Gating']

    all_clus.plot(alpha=0.5, figsize=(16, 10), marker='.', fontsize = fontsize)

    #plt.tight_layout(h_pad = 1.0) 
    #plt.subplots_adjust(top=0.2)
    
    if printed_cluster_name in(['synechococcus', 'prochlorococcus']):
        plt.title(title_number[idx] + '$\it{' + printed_cluster_name.capitalize() + '}$', fontsize = fontsize)
        plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        
    else:
        plt.title(title_number[idx] +  printed_cluster_name.capitalize(), fontsize = fontsize)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)

    if printed_cluster_name == 'picoeukaryote':
        plt.legend(all_clus.columns, fontsize = 22)
    else:
        plt.legend().remove()
          
    plt.xlabel(xlabel = 'time', fontsize = fontsize)
    plt.ylabel(ylabel = 'Counted particles', fontsize = fontsize)

    plt.savefig('C:/Users/rfuchs/Desktop/pred4_P1/' + cluster_name + '.png')   

        
###################################################################################################################
# Visualize the predictions made on SSLAMM (trained with SSLAMM data)
###################################################################################################################

import fastparquet as fp
from pred_functions import predict
from viz_functions import plot_2D

from keras.models import load_model

acq_name = 'Pulse6_2019-10-01 15h59'

folder = 'C:/Users/rfuchs/Documents/cyto_classif'
file = 'SSLAMM_L2/Labelled_' + acq_name + '.parq'


date_regex = "(Pulse[0-9]{1,2}_20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}(?:u|h)[0-9]{2})"
pred_file = acq_name + '.csv'
os.chdir(folder)

tn = pd.read_csv('train_test_nomenclature_SSLAMM.csv')
tn.columns = ['Particle_class', 'label']

# Load pre-trained model
LottyNet = load_model('C:/Users/rfuchs/Documents/cyto_classif/LottyNet_SSLAMM_focal',\
                      custom_objects={'categorical_focal_loss_fixed': CB_loss(sample_per_class)}) 

# Making formated predictions 
source_path = folder + '/' + file
dest_folder = folder
predict(source_path, folder, LottyNet, tn)

# Getting those predictions
preds = pd.read_csv(folder + '/' + pred_file)

np.mean(preds['True FFT id'] == preds['Pred FFT id'])
print(confusion_matrix(preds['True FFT id'] , preds['Pred FFT id']))
print('Macro accuracy is', precision_score(preds['True FFT id'], preds['Pred FFT id'], average='macro'))

colors = ['#96ceb4', '#ffeead', '#ffcc5c', '#ff6f69', '#588c7e', '#f2e394', '#f2ae72', '#d96459']


X = preds[['Total FLR', 'Total FLO']].values
y = preds['Pred FFT Label'].values
labels = preds['Pred FFT id'].values
z = labels[..., np.newaxis]



fig, ax = plt.subplots()
CS = ax.contour(X, labels, Z)


cmap = plt.get_cmap('Paired')
fig, ax = plt.subplots()
ax.contourf(X[:,0][..., np.newaxis], X[:,1][..., np.newaxis], z, cmap=cmap, alpha=0.5)


#####################
# 2D plots
#####################

plot_2D(preds, tn, 'Total FLR', 'Total FLO', loc = 'upper left') # FLO vs FLR
plot_2D(preds, tn, 'Total FWS', 'Total FLR', loc = 'upper left')
plot_2D(preds, tn, 'Total SWS', 'Total FLR', loc = 'upper left')
plot_2D(preds, tn, 'Total SWS', 'Total FWS', loc = 'upper left')


# Decision boundaries plot
plot_decision_boundaries2(preds, tn, 'Total FWS', 'Total FLR', \
                          loc = 'upper left', title = None, colors = None)

#===========================================
# Viz one of the SSLAMM files
#============================================

cluster_classes = ['airbubble', 'cryptophyte', 'hsnano', 'microphytoplancton',
       'nanoeucaryote', 'picoeucaryote', 'prochlorococcus',
       'synechococcus', 'unassigned particle']

source = 'C:/Users/rfuchs/Documents/cyto_classif/SSLAMM_L2'

# Extract the test dataset from full files
X_test_SLAAMM, seq_len_list_test_SLAAMM, y_test_SLAAMM, pid_list_test_SLAAMM, file_name_test_SLAAMM, le_test = gen_dataset(source, \
                            cluster_classes, [], None, nb_obs_to_extract_per_group = 100, \
                            to_balance = False, to_undersample = False, seed = None)

X = pd.DataFrame(trapz(X_test_SLAAMM, axis = 1), \
                         columns = ['SWS','Total FWS', 'FL Orange', 'Total FLR', 'Curvature'])
y = y_test_SLAAMM.argmax(1)
plot_2Dcyto(X,y , tn, 'Total FWS', 'Total FLR', colors = None)


#===========================================
# Viz of the matrix and curves
#============================================

path = r'C:\Users\rfuchs\Documents\SSLAMM_P1\SSLAMM_L3\X_train.npy'
X = np.load(path)
plt.figure(figsize = (8, 2))
plt.plot(X[2])
plt.legend(['FWS',	'SWS', 'FL Orange', 'FL Red', 'Curvature'])
plt.xlim = [1,120]
plt.show()

plt.imshow(X[2].T, aspect = 3.7)
plt.xticks([0, 39, 79, 119], ['0', '40', '80', '120'])
plt.yticks([0, 1, 2, 3, 4], ['FWS',	'SWS', 'FL Orange', 'FL Red', 'Curvature'])
plt.xlim = [1,5]
plt.ylim = [1,120]
plt.show()
