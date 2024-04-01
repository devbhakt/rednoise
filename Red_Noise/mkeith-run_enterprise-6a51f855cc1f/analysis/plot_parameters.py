#!/usr/bin/env python
'''Read pulsar glitches solution files and plot the correlation between parameters.
    Written by Y.Liu (yang.liu-50@postgrad.manchester.ac.uk).'''

#import argparse
#import sys, os
#import subprocess
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
import numpy as np
#import numpy.ma as ma
import pandas as pd
import seaborn as sns
#import pulsar_glitch as pg
#from __future__ import print_function
#from uncertainties import ufloat
#from scipy.optimize import curve_fit
#from scipy import linalg
#from astropy import coordinates as coord
#from astropy import units as u
#from matplotlib.ticker import FormatStrFormatter
#from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#from mpl_toolkits.axes_grid.inset_locator import inset_axes

sum_table = pd.read_csv('summary.csv')
def GLF1instant(row):
    GLF1ins = row['GLF1']
    if row['GLTD'] > 0:
        GLF1ins += row['GLF0D']/row['GLTD']
        if row['GLTD2'] > 0:
            GLF1ins += row['GLF0D2']/row['GLTD2']
            if row['GLTD3'] > 0:
                GLF1ins += row['GLF0D3']/row['GLTD3']
    return GLF1ins
sum_table['GLF1(instant)'] = sum_table.apply(lambda row: GLF1instant(row), axis=1)

sum_table = sum_table[sum_table['Pulsar name'] != 'J2229+6114']
print(sum_table.info(verbose=True))
#print(sum_table)

full_index = ['Pulsar name', 'Glitch No.', 'F0', 'F0 std', 'F1', 'F1 std', 'F2', 'F2 std', 
            'P', 'P std', 'P_dot', 'P_dot std', 'P_ddot', 'P_ddot std', 'E_dot', 'Tau_c', 'B_sur', 'Braking index', 
            'GLEP', 'GLEP std', 'GLF0', 'GLF0 std', 'GLF1', 'GLF1 std', 'GLF2', 'GLF2 std', 
            'GLF0(instant)', 'GLF0(instant) std', 'GLF1(instant)', 'GLF0(T=tau_g)', 'GLF0(T=tau_g) std', 
            'Tau_g', 'Waiting time', 'GLF0D', 'GLF0D std', 'GLTD', 'GLTD std', 
            'GLF0D2', 'GLF0D2 std', 'GLTD2', 'GLTD2 std', 'GLF0D3', 'GLF0D3 std', 'GLTD3', 'GLTD3 std', 
            'TNRedAmp', 'TNRedAmp std', 'TNRedGam', 'TNRedGam std']
full_index.remove('GLF0D')
full_index.remove('GLF0D2')
full_index.remove('GLF0D3')
full_index.remove('GLF0D std')
full_index.remove('GLF0D2 std')
full_index.remove('GLF0D3 std')
full_index.remove('GLTD')
full_index.remove('GLTD2')
full_index.remove('GLTD3')
full_index.remove('GLTD std')
full_index.remove('GLTD2 std')
full_index.remove('GLTD3 std')

melt_table = sum_table.drop(['GLF0D std', 'GLF0D2 std', 'GLF0D3 std', 'GLTD std', 'GLTD2 std', 'GLTD3 std'], axis=1)
melt_table = melt_table.rename({'GLF0D': 'GLF0D1', 'GLTD': 'GLTD1'}, axis=1)
melt_table = pd.wide_to_long(melt_table, ['GLF0D', 'GLTD'], i=full_index, j='Recoveries')
melt_table = melt_table.reset_index()
print(melt_table)

families = {'B1757-24':'long', 'B1830-08':'long', 'B1951+32':'long', 'B2334+61':'long', 'J1737-3137':'long', 
            'J1740+1000':'long', 'J1809-1917':'long', 'J1841-0345':'long', 'J1909+0912':'long', 
            'B1727-33':'short', 'B1757-24':'short', 'B1800-21':'short', 'B1823-13':'short', 'B1853+01':'short', 
            'J2021+3651':'short', 'J0631+1036':'none', 'J1837-0604':'none', 'J1841-0524':'none', 
            'J1850-0026':'none', 'J1907+0631':'none', 'B1930+22':'others', 'J0729-1448':'others', 
            'J1856+0245':'others', 'J2229+6114':'special'}
melt_table['family'] = melt_table['Pulsar name'].map(families)
melt_table['GLF0D'] = melt_table['GLF0D'].fillna(0)
melt_table['GLTD'] = melt_table['GLTD'].fillna(0)
#group_table = melt_table.groupby('family')
#sort_table = melt_table.sort_values(by=['GLTD'])
#print(sort_table)

sns.set_theme()
plt.figure(figsize=(12, 8))
#fig = sns.relplot(data=melt_table, x='E_dot', y='GLTD', col='family', hue='Pulsar name', style='Glitch No.', 
#                  size='Recoveries', kind='scatter')
fig = sns.relplot(data=melt_table, x='E_dot', y='GLTD', hue='family', style='Glitch No.', 
                  size='Recoveries', kind='scatter')
fig.add_legend(frameon=False)
fig.legend.set_bbox_to_anchor((1.01, 1))
plt.xscale('log')

paras = ['F0', 'F1', 'F2', 'P', 'P_dot', 'P_ddot', 'Tau_c', 'B_sur', 'Braking index', 'GLF0', 'GLF1', 
         'GLF0(instant)', 'GLF1(instant)', 'GLF0(T=tau_g)', 'Waiting time']

sns.set_theme()
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(x='E_dot', y='GLTD', data=melt_table, ax=ax, hue='family', style='Glitch No.', size='Recoveries')
ax.legend(bbox_to_anchor=(1.01, 1), loc=0, borderaxespad=0., title='PSR')
ax.set_xscale('log')
ax.set_ylabel('Time scale of recoveries')
plt.savefig('Parameters_plots/logE_dot--recoveries_time_scales.pdf')

for xpara in paras:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x=xpara, y='GLTD', data=melt_table, ax=ax, hue='family', style='Glitch No.', size='Recoveries')
    ax.legend(bbox_to_anchor=(1.01, 1), loc=0, borderaxespad=0., title='PSR')
    ax.set_ylabel('Time scale of recoveries')
    plt.savefig('Parameters_plots/'+xpara+'--recoveries_time_scales.pdf')

sns.set_theme()
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(x='E_dot', y='GLF0D', data=melt_table, ax=ax, hue='family', style='Glitch No.', size='Recoveries')
ax.legend(bbox_to_anchor=(1.01, 1), loc=0, borderaxespad=0., title='PSR')
ax.set_xscale('log')
ax.set_ylabel('Amplitude of recoveries')
plt.savefig('Parameters_plots/logE_dot--recoveries_amplitude.pdf')

for xpara in paras:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x=xpara, y='GLF0D', data=melt_table, ax=ax, hue='family', style='Glitch No.', size='Recoveries')
    ax.legend(bbox_to_anchor=(1.01, 1), loc=0, borderaxespad=0., title='PSR')
    ax.set_ylabel('Amplitude of recoveries')
    plt.savefig('Parameters_plots/'+xpara+'--recoveries_amplitude.pdf')

sns.set_theme()
for ypara in paras:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x='E_dot', y=ypara, data=melt_table, ax=ax, hue='family', style='Glitch No.', size='Recoveries')
    ax.legend(bbox_to_anchor=(1.01, 1), loc=0, borderaxespad=0., title='PSR')
    ax.set_xscale('log')
    plt.savefig('Parameters_plots/logE_dot--'+ypara+'.pdf')

y_paras = ['GLF0', 'GLF1', 'GLF0(instant)', 'GLF1(instant)', 'GLF0(T=tau_g)', 'Waiting time']

sns.set_theme()
for xpara in paras:
    for ypara in y_paras:
        if xpara != ypara:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(x=xpara, y=ypara, data=melt_table, ax=ax, hue='family', style='Glitch No.', size='Recoveries')
            ax.legend(bbox_to_anchor=(1.01, 1), loc=0, borderaxespad=0., title='PSR')
            plt.savefig('Parameters_plots/'+xpara+'--'+ypara+'.pdf')
