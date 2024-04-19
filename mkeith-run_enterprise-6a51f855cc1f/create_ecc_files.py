#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import scipy.optimize

current_dir = os.getcwd()
current_dir

infolder = current_dir.split('/')[-1]
infolder

print('\n*** This should take a while, but it should only be ran once. ***\n')

# creating an empty folder `E_e_5dig` only if it doesn't exist.

if  infolder == 'run_enterprise':
    if not os.path.isdir('model_components/E_e_5dig'):
        print('\nMaking folder `model_components/E_e_5dig`')
        os.mkdir('model_components/E_e_5dig')
    else:
        print('! Folder `E_e_5dig` already exists in `model_components/`\n')

    E_folder = 'model_components/E_e_5dig'

elif infolder == 'model_components':
    if not os.path.isdir('E_e_5dig'):
        print('\nMaking folder `E_e_5dig`')
        os.mkdir('E_e_5dig')
    else:
        print('! Folder `E_e_5dig` already exists here`\n')

    E_folder = 'E_e_5dig'

else:
    print('!! Run this inside `run_enterprise` or `run_enterprise/model_components`! Terminating script.\n')

    E_folder = ''

    exit()


eccs = np.linspace(0,0.9,90001)

x=(np.logspace(-3,0.5,1000))%(2*np.pi)
## We sample very densely close to zero, where highly eccentric orbits go wild
## Also make sure to cover the input range of data.
sampled_mean_anom = np.concatenate((x,2*np.pi-x,[1e-15,2*np.pi]))
sampled_mean_anom.sort()


count_10000 = 0 # just for printing messages

for e, i in zip(eccs, range(len(eccs))):
    
    if i % 10000 == 0:
        if count_10000 == 0:
            print('\nWriting the 1st 10000 files...\n')
        elif count_10000 == 1:
            print('\nWriting the 2nd 10000 files...\n')
        elif count_10000 == 2:
            print('\nWriting the 3rd 10000 files...\n')
        else:
            print('\nWriting the %ith 10000 files...\n' %(count_10000+1))
        count_10000 += 1
    
    ## Sample the mean anomaly space in such a way as to minimise deviations
    sampled_E = np.zeros_like(sampled_mean_anom)
    def ecc_anom(E,e,M):
        return (E-e*np.sin(E))-M

    for i in range(len(sampled_E)):
        v1=ecc_anom(0,e,sampled_mean_anom[i])
        E1=0
        for trial in np.linspace(0,2*np.pi,8)[1:]:
            E2=trial
            v2 = ecc_anom(trial,e,sampled_mean_anom[i])
            if v1*v2 < 0:
                break
            v1=v2
            E1=E2

        sol = scipy.optimize.root_scalar(ecc_anom,args=(e,sampled_mean_anom[i]),bracket=[E1,E2])
        sampled_E[i] = sol.root
    
    np.save(E_folder+'/E_e=%.5f.npy' %e, [sampled_mean_anom,sampled_E])




