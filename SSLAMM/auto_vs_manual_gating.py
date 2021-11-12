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

code_dir = 'C:/Users/rfuchs/Documents/GitHub/'
XP_dir = r'C:/Users/rfuchs/Documents/These/Oceano/XP_bias_SSLAMM/'

os.chdir(code_dir)
from phyto_curves_reco.dataset_preprocessing import homogeneous_cluster_names

os.chdir(XP_dir)

flr_regex = 'FLR([0-9]{1,2})'
datetime_regex = "(20[0-9]{2}-[0-9]{2}-[0-9]{2} [0-9]{2}h[0-9]{2})"

new_nomenclature = {'synechococcus':'ORGPICOPRO', 'microphytoplancton':'MICRO',\
                    'nanoeucaryote': 'REDNANO', 'cryptophyte': 'ORGNANO', 
                    'picoeucaryote': 'REDPICOEUK', 'prochlorococcus': 'REDPICOPRO',\
                    'Unassigned Particles': 'Unassigned Particles'}
    
pfgs = ['MICRO', 'ORGNANO', 'ORGPICOPRO', 'REDNANO', 'REDPICOEUK',\
             'REDPICOPRO']
 
spe_extract = {'MICRO': 6, 'ORGNANO': 6, 'ORGPICOPRO': 25, 'REDNANO': 6, 'REDPICOEUK': 6,\
             'REDPICOPRO': 25}

#===========================
# Import manual data 
#===========================
manual = pd.read_csv('Manual/manual_gating.csv', sep = ';', engine = 'python',\
                     usecols = ['Filename', 'set', 'count']) 

# Filter useless classes
legalClasses = ['ORGMICRO', 'ORGNANO', 'ORGPICOPRO', 'MICRO',\
    'REDNANO', 'REDPICOEUK', 'REDPICOPRO', 'inf1âmicrom_unidentified_particle',\
        'sup1âmicrom_unidentified_particle']
    
# Make the pfg categories comparable
manual = manual[['Filename', 'set', 'count']]
manual.columns = ['Filename', 'cluster', 'Count']

# Replace names of the PFG    
manual = homogeneous_cluster_names(manual)

for oldname, newname in new_nomenclature.items():
    manual['cluster'] = manual['cluster'].str.replace(oldname, newname)
       
manual = manual[manual['cluster'].isin(legalClasses)]

# Pivot the table 
manual = pd.pivot_table(manual, values='Count', index=['Filename'],
                    columns=['cluster'], aggfunc=np.sum)

manual['Unassigned Particles'] = manual['sup1âmicrom_unidentified_particle'] + \
    + manual['inf1âmicrom_unidentified_particle']
    
del(manual['inf1âmicrom_unidentified_particle'])
del(manual['sup1âmicrom_unidentified_particle'])

manual['date'] = [pd.to_datetime(re.search(datetime_regex, idx).group(1)) for idx in  manual.index]
manual.index = manual.index.str.replace('SSLAMM_FLR25', 'SSLAMM-FLR25')

manual = manual.reset_index().set_index('date')
manual['FLR'] =  [int(re.search(flr_regex, file).group(1)) for file in manual['Filename']] 


# Special extract
for pfg, flr in spe_extract.items():
    manual.loc[manual['FLR'] == flr, pfg] = np.nan


manual.to_csv('../XP_bias_both/manual_SSLAMM.csv')


#===========================
# Import automatic data
#===========================

auto = pd.read_csv('preds20211027_full_SSLAMM.csv', parse_dates = ['date'])
auto['date'] = [pd.to_datetime(re.search(datetime_regex, idx).group(1)) for idx in auto['file']]

# Total noise computation
auto = auto.rename(columns = {'file': 'Filename'})
auto['Unassigned Particles'] = auto['inf1microm'] + auto['sup1microm']


# Format the fileName to match the manual gating names
auto['Filename'] = auto['Filename'].str.replace('.parq', '.cyz')
auto['Filename'] = auto['Filename'].str.replace('Pulse', 'SSLAMM-FLR')
auto['Filename'] = auto['Filename'].str.replace('_2019', ' 2019')


# Special extract
for pfg, flr in spe_extract.items():
    auto.loc[auto['FLR'] == flr, pfg] = np.nan


# Format the name of the PFGs
auto = auto[['MICRO', 'ORGNANO', 'ORGPICOPRO', 'REDNANO', 'REDPICOEUK',\
             'REDPICOPRO', 'Unassigned Particles',\
             'Filename', 'date']]

auto.set_index('date', inplace = True) 
auto = auto.sort_index()
auto.to_csv('../XP_bias_both/auto_SSLAMM.csv')


#===========================
# Join plot of the series
#===========================

fig, axs = plt.subplots(3, 2, figsize= (10, 10))

pfgs = ['MICRO', 'ORGNANO', 'ORGPICOPRO', 'REDNANO', 'REDPICOEUK', 'REDPICOPRO']

for idx, pfg in enumerate(pfgs):
    # Compute the correlation
    data = auto[[pfg]].join(manual[pfg], lsuffix = '_auto', rsuffix = '_manual') 
    print(data.corr())
            
    data = data.dropna()
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
    
    if len(X) == 0:
        continue
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

pfg = 'MICRO'
data = auto[[pfg]].join(manual[pfg], lsuffix = '_auto', rsuffix = '_manual') 
data = data.dropna()
data[pfg + '_manual'].sort_values()[-8:]
manual[pfg].max()

data.loc[data['MICRO_manual'].sort_values().index[-1]]['REDPICOPRO_auto']
data.loc['2021-03-02 18:11:00']


dd = manual.loc[['2021-03-02 18:11:00',\
                               '2021-01-09 08:11:00',\
                               '2021-01-24 16:11:00']].T

ee = auto.loc[['2021-03-02 18:11:00',\
               '2021-01-09 08:11:00',\
               '2021-01-24 16:11:00']].T
dd[dd.columns[-1]]
    


            

