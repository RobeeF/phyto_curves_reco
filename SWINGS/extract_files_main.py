# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:27:32 2019

@author: Utilisateur
"""

import os
os.chdir('C:/Users/rfuchs/Documents/GitHub/phyto_curves_reco')
from from_cytoclus_to_curves_values import extract_non_labeled_curves


##################################################################################################
# SWINGS Unlabelled
##################################################################################################


data_source = 'C:/Users/rfuchs/Documents/These/Oceano/XP_bias_SWINGS/full_data/L0'
data_destination = 'C:/Users/rfuchs/Documents/These/Oceano/XP_bias_SWINGS/full_data/L1'
flr_num = 20 # And 6
extract_non_labeled_curves(data_source, data_destination, flr_num = flr_num)
flr_num = 5 # And 6
extract_non_labeled_curves(data_source, data_destination, flr_num = flr_num)

