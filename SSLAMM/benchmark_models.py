# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:19:22 2020

@author: rfuchs
"""

import re
import os 
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, accuracy_score
#from imblearn.under_sampling import EditedNearestNeighbours

os.chdir('C:/Users/rfuchs/Documents/GitHub/phyto_curves_reco')

# Load nomenclature
tn = pd.read_csv('train_test_nomenclature.csv')
tn.columns = ['Particle_class', 'label']

############################################################
# Training of other algorithms on the unbiased dataset
############################################################
import fastparquet as fp

from imblearn.under_sampling import RandomUnderSampler

from keras import load_model
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier

from sklearn.preprocessing import OneHotEncoder

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

cluster_classes = ['airbubble', 'cryptophyte', 'nanoeucaryote',\
                   'inf1microm_unidentified_particle', 'microphytoplancton',\
                'picoeucaryote', 'prochlorococcus', \
                'sup1microm_unidentified_particle', 'synechococcus']
    
        
def clf_eval(params):
    ''' Wrapper around classifiers for them to be fed into hyperopt '''
    classif = params['classif']
    del params['classif']
    
    if classif == 'knn':
        print('knn')
        clf = KNeighborsClassifier(**params)
    elif classif == 'svm':
        print('svm')
        clf = svm.SVC(**params)
    elif classif == 'lgbm':
        print('lgbm')
        clf = LGBMClassifier(**params)
        
    clf.fit(X_train, y_train)
    pred_valid = clf.predict(X_valid)
    accuracy = accuracy_score(y_valid, pred_valid)
    
    return {'loss': -accuracy, 'status': STATUS_OK}

def prec_rec_function(y_test, preds, cluster_classes, algo):
    ''' Compute the precision and recall for all classes'''
    prec = precision_score(y_test, preds, average=None)
    prec = dict(zip(cluster_classes, prec))
    prec['algorithm'] = algo
    
    recall= recall_score(y_test, preds, average=None)
    recall = dict(zip(cluster_classes, recall))
    recall['algorithm'] = algo
    
    return prec, recall

#************************************
# Data importation 
#************************************

list_dir = 'C:/Users/rfuchs/Documents/cyto_classif/XP_Listmodes'
list_files = os.listdir(list_dir)
parts_to_keep_files = [file for file in list_files if re.search('.csv', file)]
list_files = [file for file in list_files if re.search('.parq', file)]

# Import the list of the selected particles
pids_df = pd.DataFrame()
for file in parts_to_keep_files:
    df = pd.read_csv(list_dir + '/' + file)
    pids_df = pids_df.append(df)

#==============================================
# Build the train and test set
#==============================================

# Train set 
train_particles = pd.read_csv(list_dir + '/' + 'train_pids.csv')
train_files = np.unique(train_particles['acq'])
train = pd.DataFrame()

for file in train_files:
    pf = fp.ParquetFile(list_dir + '/' + file)
    all_particles = pf.to_pandas()
    
    pp = list(all_particles['Particle ID'])
    
    pids = list(pids_df[pids_df['acq'] == file]['Particle ID'])
    selected_particles = all_particles.set_index('Particle ID').loc[pids]
    
    train = train.append(selected_particles)

# Valid set 
valid = pd.DataFrame()
valid_particles = pd.read_csv(list_dir + '/' + 'valid_pids.csv')
valid_files = np.unique(valid_particles['acq'])


for file in valid_files:
    pf = fp.ParquetFile(list_dir + '/' + file)
    all_particles = pf.to_pandas()
    
    pids = list(pids_df[pids_df['acq'] == file]['Particle ID'])
    selected_particles = all_particles.set_index('Particle ID').loc[pids]
    
    valid = valid.append(selected_particles)

# Test test
test_particles = pd.read_csv(list_dir + '/' + 'test_pids.csv')
test_files = np.unique(test_particles['acq'])
test = pd.DataFrame()

for file in test_files:
    pf = fp.ParquetFile(list_dir + '/' + file)
    all_particles = pf.to_pandas()
    
    pids = list(pids_df[pids_df['acq'] == file]['Particle ID'])
    selected_particles = all_particles.set_index('Particle ID').loc[pids]
    
    test = test.append(selected_particles)

# Train / testsplit
X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]

X_valid = valid.iloc[:, :-1]
y_valid = valid.iloc[:, -1]

X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]


# Delete empty columns
X_train = X_train.iloc[:, X_train.columns != 'Curvature Center of gravity']
X_train = X_train.iloc[:, X_train.columns != 'Curvature Asymmetry']

X_valid = X_valid.iloc[:, X_valid.columns != 'Curvature Center of gravity']
X_valid = X_valid.iloc[:, X_valid.columns != 'Curvature Asymmetry']

X_test = X_test.iloc[:, X_test.columns != 'Curvature Center of gravity']
X_test = X_test.iloc[:, X_test.columns != 'Curvature Asymmetry']

p = X_train.shape[1]

# RUS for cleaning data (will be changed to SMOTEEN)
ss = dict()
for cc in cluster_classes:
    ss[cc] = min((y_train == cc).sum() + (y_valid == cc).sum(), 2000)
    
rus = RandomUnderSampler(sampling_strategy = ss)
    
X_tv, y_tv = rus.fit_resample(X_train.append(X_valid), y_train.append(y_valid)) 

np.unique(y_valid, return_counts = True)

enc = OneHotEncoder(handle_unknown='ignore',  sparse = False)
y_train_oh = enc.fit_transform(y_train.values.reshape(-1, 1),)
y_valid_oh = enc.fit_transform(y_valid.values.reshape(-1, 1))
y_test_oh = enc.fit_transform(y_test.values.reshape(-1, 1))

y_train

os.chdir('C:/Users/rfuchs/Documents/cyto_classif')

np.savez_compressed('XP_Listmodes/train', X = X_train, y = y_train_oh)
np.savez_compressed('XP_Listmodes/test', X = X_test, y = y_test_oh)
np.savez_compressed('XP_Listmodes/valid', X = X_valid, y = y_valid_oh)
    

X_valid.isna().sum().sum()
#************************************
# Looking for the best hyperparams 
#************************************
#from sklearn.model_selection import GridSearchCV
algo=tpe.suggest
nb_evals = 50

# kNN
nn = (1, 2, 3, 4, 5, 6, 7)
w = ('uniform','distance')
algs = ('ball_tree', 'kd_tree', 'brute')
p_knn = (1, 2, 3)

knn_params = {'classif': 'knn', 'n_neighbors': hp.choice('n_neighbors', nn), 
               'weights': hp.choice('weights', w),
               'algorithm': hp.choice('algorithm', algs),\
                'p': hp.choice('p', p_knn)}


knn_best = fmin(
    fn=clf_eval, 
    space=knn_params,
    algo=algo,
    max_evals = nb_evals)


# SVM
kernel = ('rbf', 'linear')
gamma = (1e-3, 1e-4)
C = (1, 10, 100, 1000)
class_names, nb_samples  =np.unique(y_train, return_counts = True)
reweighted = dict(zip(class_names, 1/ nb_samples))
equal_weights = dict(zip(class_names, np.full(len(class_names), 1 / len(class_names))))
class_weight = (reweighted, equal_weights)

svm_params = {'classif': 'svm',\
            'kernel': hp.choice('kernel', kernel),\
            'gamma': hp.choice('gamma', gamma),
            'C': hp.choice('C', C),\
            'class_weight': hp.choice('class_weight', class_weight)}

svm_best = fmin(
    fn=clf_eval, 
    space=svm_params,
    algo=algo,
    max_evals = nb_evals)


# Lgbm
lr = (0.005, 0.01)
n_est = (8,16,24)
num_leaves = (6,8,12,16)
bt = ('gbdt', 'dart')
objective = ('binary')
max_bin = (255, 510)
colsample_bytree = (0.64, 0.65, 0.66)
subsample = (0.7,0.75)
reg_alpha = (1,1.2)
reg_lambda = (1,1.2,1.4)
is_unbalance = (True, False)

lgbm_params = {
    'classif': 'lgbm',
    'learning_rate': hp.choice('learning_rate', lr),
    'n_estimators': hp.choice('n_estimators', n_est),
    'num_leaves': hp.choice('num_leaves', num_leaves), # large num_leaves helps improve accuracy but might lead to over-fitting
    'boosting_type': hp.choice('boosting_type', bt), # for better accuracy -> try dart
    'objective': hp.choice('objective', objective),
    'max_bin': hp.choice('max_bin', max_bin), # large max_bin helps improve accuracy but might slow down training progress
    'colsample_bytree': hp.choice('colsample_bytree', colsample_bytree),
    'subsample': hp.choice('subsample', subsample),
    'reg_alpha': hp.choice('reg_alpha', reg_alpha),
    'reg_lambda':  hp.choice('reg_lambda', reg_lambda),
    'is_unbalance': hp.choice('is_unbalance', is_unbalance)
    }

lgbm_best = fmin(
    fn=clf_eval, 
    space=lgbm_params,
    algo=algo,
    max_evals = nb_evals)


#********************************
# Fitting of the models
#********************************

# KNN
knn = KNeighborsClassifier(n_neighbors = nn[knn_best['n_neighbors']], \
                           weights = w[knn_best['weights']], \
                               algorithm = algs[knn_best['algorithm']],
                               p = p_knn[knn_best['p']])
knn.fit(X_train, y_train)


# SVM
svm = svm.SVC(kernel = kernel[svm_best['kernel']],\
              gamma = gamma[svm_best['gamma']],
               C = C[svm_best['C']])
svm.fit(X_train, y_train)

# LGBM
lgbm = LGBMClassifier(learning_rate = lr[lgbm_best['learning_rate']],
    n_estimators = n_est[lgbm_best['n_estimators']],
    num_leaves = num_leaves[lgbm_best['num_leaves']], # large num_leaves helps improve accuracy but might lead to over-fitting
    boosting_type = bt[lgbm_best['boosting_type']], # for better accuracy -> try dart
    objective = objective[lgbm_best['objective']],
    max_bin = max_bin[lgbm_best['max_bin']], # large max_bin helps improve accuracy but might slow down training progress
    colsample_bytree = colsample_bytree[lgbm_best['colsample_bytree']],
    subsample = subsample[lgbm_best['subsample']],
    reg_alpha = reg_alpha[lgbm_best['reg_alpha']],
    reg_lambda = reg_lambda[lgbm_best['reg_lambda']],
    is_unbalance = is_unbalance[lgbm_best['is_unbalance']])

lgbm.fit(X_train, y_train)

# FFNN
ffnn = load_model('trained_models/XP_ffnn')

# LottyNet !
cnn = load_model('trained_models/XP_cnn')

#********************************
# Prediction of the models
#********************************

knn_preds = knn.predict(X_test)  
svm_preds = svm.predict(X_test)  
lgbm_preds = lgbm.predict(X_test) 
ffnn_preds = ffnn.predict(X_test) 
ffnn_preds = enc.inverse_transform(ffnn_preds)
cnn_preds = cnn.predict(X_test) 
cnn_preds = enc.inverse_transform(cnn_preds)

#********************************
# Accuracy computations
#********************************

prec = pd.DataFrame(columns= cluster_classes + ['algorithm'])
recall = pd.DataFrame(columns= cluster_classes + ['algorithm'])


# KNN 
prec_knn, recall_knn = prec_rec_function(y_test, knn_preds, cluster_classes, 'knn')
prec = prec.append(prec_knn, ignore_index = True)
recall = recall.append(recall_knn, ignore_index = True)

# SVM
prec_svm, recall_svm = prec_rec_function(y_test, svm_preds, cluster_classes, 'svm')
prec = prec.append(prec_svm, ignore_index = True)
recall = recall.append(recall_svm, ignore_index = True)


# LGBM
prec_lgbm, recall_lgbm = prec_rec_function(y_test, lgbm_preds, cluster_classes, 'lgbm')
prec = prec.append(prec_lgbm, ignore_index = True)
recall = recall.append(recall_lgbm, ignore_index = True)

# FFNN
prec_ffnn, recall_ffnn = prec_rec_function(y_test, ffnn_preds, cluster_classes, 'ffnn')
prec = prec.append(prec_ffnn, ignore_index = True)
recall = recall.append(recall_ffnn, ignore_index = True)


# CNN
prec_cnn, recall_cnn = prec_rec_function(y_test, cnn_preds, cluster_classes, 'cnn')
prec = prec.append(prec_cnn, ignore_index = True)
recall = recall.append(recall_cnn, ignore_index = True)


#********************************
# Final output
#********************************

prec_recall_comp = pd.concat([prec, recall], axis = 1)
prec_recall_comp.columns = [cc + '_prec' for cc in cluster_classes] + \
                ['algorithm_prec'] + [cc + '_recall' for cc in cluster_classes] +\
                    ['algorithm']

prec_recall_comp = prec_recall_comp.drop('algorithm_prec', 1)

prec_recall_comp.set_index('algorithm').to_csv('prec_recall_comp.csv')
