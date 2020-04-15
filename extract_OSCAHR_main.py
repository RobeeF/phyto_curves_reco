# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:27:32 2019

@author: Utilisateur
"""
import os
os.chdir('C:/Users/Utilisateur/Documents/GitHub/planktonPipeline/extract_Pulse_values')
from from_cytoclus_to_curves_values import extract_Oscar

# Extract the FLR 5
listmode_source = 'Z:/CS-68-2015/OSCAHR/oscahr/L1-OSCAHR/process_January_2017_OSCAHR_PM/process_12-01-2017_flr5_all/listmode_OSCAHR_flr5_all_January2017'
pulse_source = 'Z:/CS-68-2015/OSCAHR/oscahr/L1-OSCAHR/OSCAHR-pulseshapedefault'
data_destination = 'C:/Users/Utilisateur/Documents/GitHub/planktonPipeline/extract_Pulse_values/data/L2_OSCAHR'

extract_Oscar(listmode_source, pulse_source, data_destination, flr_num = 5, is_untreated = True)

# Extract the FLR 30
listmode_source = 'Z:/CS-68-2015/OSCAHR/oscahr/L1-OSCAHR/process_January_2017_OSCAHR_PM/process_16-01-2017_FLR30_all/listmode_OSCAHR_FLR30_all_January2017'
extract_Oscar(listmode_source, pulse_source, data_destination, flr_num = 30, is_untreated = True)