# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:27:32 2019

@author: Utilisateur
"""
import os
os.chdir('C:/Users/rfuchs/Documents/GitHub/planktonPipeline/extract_Pulse_values')
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

data_source = 'C:/Users/rfuchs/Documents/SSLAMM_P1/SSLAMM_lab_uncompiled_L1'
data_destination = 'C:/Users/rfuchs/Documents/SSLAMM_P1/SSLAMM_lab_compiled_L1'
flr_num = 25
extract_labeled_curves(data_source, data_destination, flr_num = flr_num, spe_extract_FLR = True)
