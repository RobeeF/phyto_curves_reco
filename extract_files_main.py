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

data_source = 'C:/Users/rfuchs/Documents/SSLAMM_P2/SSLAMM-P1-pulseshape_2'
data_destination = 'C:/Users/rfuchs/Documents/SSLAMM_P2/SSLAMM_L1'
flr_num = 25 # And 6

extract_non_labeled_curves(data_source, data_destination, flr_num = flr_num)

#====================
# P2 
#====================

data_source = 'E:/SSLAMMP2-defaultpulseshapes'
data_destination = 'C:/Users/rfuchs/Documents/SSLAMM_P2'
flr_num = 6 # And 6

extract_non_labeled_curves(data_source, data_destination, flr_num = flr_num)

##################################################################################################
# ENDOUME Labelled
##################################################################################################

#data_source = 'C:/Users/rfuchs/Documents/cyto_classif/training_11sets/SSLAMM/L1'
data_source = 'C:/Users/rfuchs/Documents/cyto_classif/training_9sets/SSLAMM/L1/P2'
data_destination = 'C:/Users/rfuchs/Documents/cyto_classif/training_9sets/SSLAMM/L2/P2_lab'
flr_num = 7
extract_labeled_curves(data_source, data_destination, flr_num = flr_num, spe_extract_FLR = True)


# Manque flr25 pour P1
# Labelled_Pulse6_2019-09-20 07h59: 
# Labelled_Pulse6_2020-05-01 03h59: Que des airbubbles
# Labelled_Pulse25_2019-11-18 02h07: Moins de 6000 particules
import fastparquet as fp
pfile = fp.ParquetFile(data_destination + '/' + 'Labelled_Pulse6_2019-09-20 07h59.parq')
df = pfile.to_pandas()
df.cluster.unique()
len(df.index.unique())
