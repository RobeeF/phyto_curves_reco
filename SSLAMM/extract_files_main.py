# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:27:32 2019

@author: Utilisateur
"""

import os
os.chdir('C:/Users/rfuchs/Documents/GitHub/phyto_curves_reco')
from from_cytoclus_to_curves_values import extract_labeled_curves, extract_non_labeled_curves
from time import time

##################################################################################################
# FUMSECK
##################################################################################################

# Extract the FLR 6
data_source = 'FUMSECK-L1/FUMSECK_L1_FLR25'
data_destination = 'FUMSECK_L2'
flr_num = 6
extract_labeled_curves(data_source, data_destination, flr_num = flr_num, spe_extract_FLR = True)


# Extract the FLR 25
data_source = 'FUMSECK-L1/FUMSECK_L1_FLR25'
data_destination = 'FUMSECK_L2'
flr_num = 25
extract_labeled_curves(data_source, data_destination, flr_num = flr_num, spe_extract_FLR = True)


##################################################################################################
# ENDOUME Unlabelled
##################################################################################################

#====================
# P1 
#====================

data_source = 'C:/Users/rfuchs/Documents/These/Oceano/XP_biais/full_data/L0'
data_destination = 'C:/Users/rfuchs/Documents/These/Oceano/XP_biais/full_data/L1'
flr_num = 25 # And 6
extract_non_labeled_curves(data_source, data_destination, flr_num = flr_num)
flr_num = 6 # And 6
extract_non_labeled_curves(data_source, data_destination, flr_num = flr_num)

#====================
# P2 
#====================

data_source = 'C:/Users/rfuchs/Documents/cyto_classif/SSLAMM_P2/L0'
data_destination = 'C:/Users/rfuchs/Documents/cyto_classif/SSLAMM_P2/L1'
flr_num = 25 # And 6

extract_non_labeled_curves(data_source, data_destination, flr_num = flr_num)


#====================
# P3
#====================

data_source = 'C:/Users/rfuchs/Documents/cyto_classif/SSLAMM_P3_L0'
data_destination = 'C:/Users/rfuchs/Documents/cyto_classif/SSLAMM_P3_L1'

extract_non_labeled_curves(data_source, data_destination, flr_num = 25)
extract_non_labeled_curves(data_source, data_destination, flr_num = 6)

#====================
# P4
#====================

data_source = 'C:/Users/rfuchs/Documents/cyto_classif/P4'
data_destination = 'C:/Users/rfuchs/Documents/cyto_classif/P4_L1'

extract_non_labeled_curves(data_source, data_destination, flr_num = 25)
extract_non_labeled_curves(data_source, data_destination, flr_num = 6)

#====================
# P5
#====================

data_source = 'C:/Users/rfuchs/Documents/cyto_classif/P6/L1'
data_destination = 'C:/Users/rfuchs/Documents/cyto_classif/P6/L1'

extract_non_labeled_curves(data_source, data_destination, flr_num = 25)
extract_non_labeled_curves(data_source, data_destination, flr_num = 6)

#====================
# P6
#====================

data_source = 'C:/Users/rfuchs/Documents/cyto_classif/P6'
data_destination = 'C:/Users/rfuchs/Documents/cyto_classif/P66'

extract_non_labeled_curves(data_source, data_destination, flr_num = 25)
extract_non_labeled_curves(data_source, data_destination, flr_num = 6)


#====================
# P7
#====================

data_source = 'C:/Users/rfuchs/Documents/cyto_classif/P7_L0'
data_destination = 'C:/Users/rfuchs/Documents/cyto_classif/P7_L1'

extract_non_labeled_curves(data_source, data_destination, flr_num = 25)
extract_non_labeled_curves(data_source, data_destination, flr_num = 6)

#====================
# P8
#====================

data_source = 'C:/Users/rfuchs/Documents/cyto_classif/P8_L0'
data_destination = 'C:/Users/rfuchs/Documents/cyto_classif/P8_L1'

extract_non_labeled_curves(data_source, data_destination, flr_num = 25)
extract_non_labeled_curves(data_source, data_destination, flr_num = 6)


#====================
# P9
#====================

data_source = 'C:/Users/rfuchs/Documents/cyto_classif/P9'
data_destination = 'C:/Users/rfuchs/Documents/cyto_classif/P9_L1'

extract_non_labeled_curves(data_source, data_destination, flr_num = 25)
extract_non_labeled_curves(data_source, data_destination, flr_num = 6)


#====================
# P10
#====================

data_source = 'C:/Users/rfuchs/Documents/cyto_classif/P10_L1'
data_destination = 'C:/Users/rfuchs/Documents/cyto_classif/P10_L2'

extract_non_labeled_curves(data_source, data_destination, flr_num = 25)
extract_non_labeled_curves(data_source, data_destination, flr_num = 6)


#====================
# P11
#====================

data_source = 'C:/Users/rfuchs/Documents/cyto_classif/P11'
data_destination = 'C:/Users/rfuchs/Documents/cyto_classif/P11_L2'

extract_non_labeled_curves(data_source, data_destination, flr_num = 25)
extract_non_labeled_curves(data_source, data_destination, flr_num = 6)


#====================
# P11
#====================

data_source = 'C:/Users/rfuchs/Documents/cyto_classif/P12'
data_destination = 'C:/Users/rfuchs/Documents/cyto_classif/P12_L2'

extract_non_labeled_curves(data_source, data_destination, flr_num = 25)
extract_non_labeled_curves(data_source, data_destination, flr_num = 6)


##################################################################################################
# ENDOUME Labelled
##################################################################################################
data_source = 'C:/Users/rfuchs/Documents/cyto_classif/training_9sets/SSLAMM/L1/P2'
data_destination = 'C:/Users/rfuchs/Documents/cyto_classif/training_9sets/SSLAMM/L2/P2_lab'
flr_num = 25
extract_labeled_curves(data_source, data_destination, flr_num = flr_num, spe_extract_FLR = True)


##################################################################################################
# Piano Microphyto
##################################################################################################
data_source = 'C:/Users/rfuchs/Documents/cyto_classif/PIANO/L1'
data_destination = 'C:/Users/rfuchs/Documents/cyto_classif/PIANO/L2'
flr_num = 11
extract_labeled_curves(data_source, data_destination, flr_num = flr_num, spe_extract_FLR = True)




# Manque flr25 pour P1
# Labelled_Pulse6_2019-09-20 07h59: 
# Labelled_Pulse6_2020-05-01 03h59: Que des airbubbles
# Labelled_Pulse25_2019-11-18 02h07: Moins de 6000 particules
import fastparquet as fp
pfile = fp.ParquetFile(data_destination + '/' + 'Labelled_Pulse6_2020-02-19 05h59.parq')
df = pfile.to_pandas()
df.cluster.unique()
len(df.index.unique())


# Hack to remove:
data_source = 'C:/Users/rfuchs/Documents/cyto_classif/training_9sets/SSLAMM/L1/P2'
data_destination = 'C:/Users/rfuchs/Documents/cyto_classif/training_9sets/SSLAMM/L2/P2_lab'
import fastparquet as fp
pfile = fp.ParquetFile(data_destination + '/' + 'Labelled_Pulse25_2020-05-01 04h07.parq')
df = pfile.to_pandas()


list(df.loc[df.isna().any(1)]
indices = list(np.where(df.isna()))
np.where(np.asanyarray(np.isnan(df)))
pd.isnull(df).any().nonzero()[0]
(pd.isna(df.T) == True).any()
idx = np.isnan(df.values).any(axis=(1,2))

X_train = X_train[~idx]

##################################################################################################
# XP biais labelled
##################################################################################################
parent_repo = 'C:/Users/rfuchs/Documents/These/Oceano/XP_biais/Pulse_shapes/'
expert_repos = os.listdir(parent_repo)
flr_num = 25

for repo_name in expert_repos:
    data_source = parent_repo + repo_name
    print(repo_name)
    data_destination = 'C:/Users/rfuchs/Desktop/formatted_XP/' + repo_name
    
    extract_labeled_curves(data_source, data_destination, flr_num = flr_num)
    
##################################################################################################
# Chloé labelled files
##################################################################################################
data_source = 'C:/Users/rfuchs/Documents/cyto_classif/ChloeCaille_SSLAMMdata/'
expert_repos = os.listdir(parent_repo)
data_destination = parent_repo
flr_num = 25
    
extract_labeled_curves(data_source, data_destination, flr_num = flr_num)
 
import fastparquet as fp   
pfile = fp.ParquetFile(data_destination + '/' + 'Labelled_Pulse25_2020-10-18 06h06.parq')
total_df = pfile.to_pandas()
set(total_df.cluster)

a = homogeneous_cluster_names(total_df)
set(a.cluster)


