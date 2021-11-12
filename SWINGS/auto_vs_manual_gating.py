# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 10:37:38 2021

@author: rfuchs
"""

import re
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

code_dir = 'C:/Users/rfuchs/Documents/GitHub/phyto_curves_reco'
XP_dir = 'C:/Users/rfuchs/Documents/These/Oceano/XP_bias_SWINGS/'

os.chdir(code_dir)
from dataset_preprocessing import homogeneous_cluster_names_swings

os.chdir(XP_dir)

flr_regex = 'FLR([0-9]{1,2})'
datetime_regex = "(20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}h[0-9]{2})"

new_nomenclature = {'synechococcus':'ORGPICOPRO', 'microphytoplancton':'MICRO',\
                    'nanoeucaryote': 'REDNANO', 'cryptophyte': 'ORGNANO', 
                    'picoeucaryote': 'REDPICOEUK', 'prochlorococcus': 'REDPICOPRO',\
                    'Unassigned Particles': 'Unassigned Particles'}
    
pfgs = ['MICRO', 'ORGNANO', 'ORGPICOPRO', 'REDNANO', 'REDPICOEUK',\
             'REDPICOPRO']
 
spe_extract = {'MICRO': 5, 'ORGNANO': 5, 'ORGPICOPRO': 20, 'REDNANO': 5, 'REDPICOEUK': 5,\
             'REDPICOPRO': 20}

#===========================
# Import manual data 
#===========================

manual = pd.read_csv(XP_dir + 'Manual/SWINBGS-abundances-V1.csv', sep = ';', usecols = ['Filename', 'Set', 'Count'])

# Filter useless classes
legalClasses = ['HSNANO', 'ORGMICRO', 'ORGNANO','ORGPICOPRO_synecho', 'REDMICRO',\
    'REDNANO_nanoHFLR','REDNANO_nanoeuk', 'REDPICOEUK_picoHFLR', 'REDPICOEUK_picoeuk',\
 'Redpicopro', 'Unassigned Particles', 'bruitdefond']
    
manual = manual[manual['Set'].isin(legalClasses)]

# Make the pfg categories comparable
manual = manual[['Filename', 'Set', 'Count']]
manual.columns = ['Filename', 'cluster', 'Count']

# Replace names of the PFG
manual['cluster'] = np.where(manual['cluster'] == 'bruitdefond',\
                             'Unassigned Particles', manual['cluster'])
    
manual = homogeneous_cluster_names_swings(manual)

# Pivot the table 
manual = pd.pivot_table(manual, values='Count', index=['Filename'],
                    columns=['cluster'], aggfunc=np.sum)


manual['REDNANO'] = manual['HSNANO'] + manual['REDNANO']
manual = manual.rename(columns = {'Redpicopro': 'REDPICOPRO'})
manual = manual[['MICRO', 'ORGNANO', 'ORGPICOPRO', 'REDNANO', 'REDPICOEUK', 'REDPICOPRO',
       'Unassigned Particles']]

# Delete beads observations
manual = manual[~manual.index.str.contains('beads')]

manual['date'] = [pd.to_datetime(re.search(datetime_regex, idx).group(1)) for idx in manual.index]
manual['FLR'] =  [int(re.search(flr_regex, file).group(1)) for file in manual.index] 

manual.set_index('date', inplace = True) 
manual = manual.sort_index()


# Special extract
for pfg, flr in spe_extract.items():
    manual.loc[manual['FLR'] == flr, pfg] = np.nan


manual.to_csv('../XP_bias_both/manual_SWINGS.csv')


#===========================
# Import automatic data
#===========================

auto = pd.read_csv('preds20211027_full.csv', parse_dates = ['date'])
auto = auto[['MICRO', 'ORGNANO', 'ORGPICOPRO', 'REDNANO', 'REDPICOEUK', 'REDPICOPRO',
    'inf1microm', 'sup1microm', 'file', 'date', 'FLR']]
auto['date'] = [pd.to_datetime(re.search(datetime_regex, idx).group(1)) for idx in auto['file']]

auto = auto.fillna(0)

# Total noise computation
auto['Unassigned Particles'] = auto['inf1microm'] + auto['sup1microm']

# Format the fileName to match the manual gating names
auto['file'] = auto['file'].str.replace('.parq', '.cyz')
auto['file'] = auto['file'].str.replace('Pulse', 'MAP-IO-SWINGS-FLR')
auto['file'] = auto['file'].str.replace('_2021', ' 2021')

# Special extract
for pfg, flr in spe_extract.items():
    auto.loc[auto['FLR'] == flr, pfg] = np.nan


# Format the name of the PFGs
auto = auto[['MICRO', 'ORGNANO', 'ORGPICOPRO', 'REDNANO', 'REDPICOEUK', 'REDPICOPRO',
       'Unassigned Particles', 'file', 'date']]
auto = auto.rename(columns = {"file": 'Filename'})


auto.set_index('date', inplace = True) 
auto = auto.sort_index()

auto.to_csv('../XP_bias_both/auto_SWINGS.csv')

#===========================
# Join plot of the series
#===========================

fig, axs = plt.subplots(3, 2, figsize= (10, 10))

pfgs = ['MICRO', 'ORGNANO', 'ORGPICOPRO', 'REDNANO', 'REDPICOEUK', 'REDPICOPRO']

for idx, pfg in enumerate(pfgs):
    # Compute the correlation
    data = auto[[pfg]].join(manual[pfg], lsuffix = '_auto', rsuffix = '_manual') 
    print(data.corr())
            
    # Plot the Series
    axs[idx % 3][idx % 2].plot(data[pfg + '_manual'], label = 'Manual classification')
    axs[idx % 3][idx % 2].plot(data[pfg + '_auto'], label = 'Automatic classification')

    
    axs[idx % 3][idx % 2].set_title(pfg)
    axs[idx % 3][idx % 2].tick_params(axis='x', rotation = 30, labelsize = 10)

axs[0][0].legend()
plt.tight_layout()
plt.show()
    
    
#===========================
# Regression
#===========================

pfgs = ['MICRO', 'ORGNANO', 'ORGPICOPRO', 'REDNANO', 'REDPICOEUK', 'REDPICOPRO']

fig, axs = plt.subplots(2, 3, figsize= (12, 8))
for idx, pfg in enumerate(pfgs):
    # Compute the correlation
    data = auto[[pfg]].join(manual[pfg], lsuffix = '_auto', rsuffix = '_manual') 
    data = data.dropna()
    data['intercept'] = 1
    
    # Perform the regression
    X = data[[pfg + '_manual', 'intercept']].values
    y = data[pfg + '_auto'].values

    mod = sm.OLS(y, X, hasconst = True)
    res = mod.fit()
    #print(res.summary())
    
    # Store the params and compute the fitted values
    a = res.params[0]
    b = res.params[1]
    x = np.linspace(0, X[:,0].max())
    fitted_values = a * x + b
    
    axs[idx % 2][idx % 3].scatter(data[pfg + '_manual'], data[pfg + '_auto'])
    axs[idx % 2][idx % 3].plot(x, fitted_values)

    axs[idx % 2][idx % 3].set_title(pfg)
    axs[idx % 2][idx % 3].legend(['y = ' + str(round(b, 2)) + ' + ' +  str(round(a, 2)) + 'x\n' + \
                 'R2 = ' + str(np.round(res.rsquared, 2)) + '\n' + 'n = ' + str(len(X))],\
                    loc = 'upper left') 

    axs[idx % 2][idx % 3].set_xlabel('Manual classification')
    axs[idx % 2][idx % 3].set_ylabel('Automatic classification')
    axs[idx % 2][idx % 3].tick_params(axis='x', rotation = 30, labelsize = 10)
    axs[idx % 2][idx % 3].set_xlim([0, X[:,0].max()])
    axs[idx % 2][idx % 3].set_ylim([0, X[:,0].max()])

plt.tight_layout()



