import numpy as np
import pylab as pl
from scipy.interpolate import interp1d

from gwb_paper_plots import load_unbinned_results_new

# some good pulsars with no TN in radio or gamma:
# J1614-2230, J2043+1711 (eh), J2302+4442, J0747+6620, J2017+0603, J1741+1351, J0340+4130, J2214+3000

# NANOGrav measurements, copied in from 12.5 year paper (Alam et al)
# NB: convention is P(f) = A^2 * (f/1yr)^gamma, units on A^2 = mus^2/yr.
# 2 = in GWB sample
# 1 = in Fermi sample,
# 0 = not in Fermi sample
nanog = {
'J0030+0451' : [0.003, 6.3, 2],
'J0613-0200' : [0.123, 2.1, 2],
'J1012+5307' : [0.406, 1.6, 0],
'J1643-1224' : [1.498, 1.4, 0],
'J1713+0747' : [0.030, 1.3, 1],
'J1744-1134' : [0.155, 2.2, 1],
'J1747-4036' : [0.709, 3.3, 1],
'J1853+1303' : [0.140, 2.2, 0],
'J1857+0943' : [0.054, 3.4, 0],
'J1903+0327' : [1.482, 1.6, 0],
'J1909-3744' : [0.028, 2.7, 0],
'J1939+2134' : [0.099, 3.3, 2],
'J2145-0750' : [0.347, 2.1, 0],
'J2317+1439' : [0.007, 6.4, 1]
}

def read_nanograv():
    # convert to P(f) at period of 1 year
    rvals = dict()
    for key in nanog:
        A = nanog[key][0]
        g = nanog[key][1]
        P = A**2*1e-12
        rvals[key] = [P,nanog[key][1],nanog[key][2]]
    return rvals


# PPTA measurements, copied in Goncharov et al.
# NB values in paper are log10(A), and P(F) = A^2/12*pi^2, units of s
ppta = {
'J0437-4715' : [-14.56, 2.99, 1],
'J0613-0200' : [-14.26, 4.17, 1],
'J0711-6830' : [-13.04, 1.09, 0],
'J1024-0719' : [-14.62, 6.39, 0],
'J1600-3053' : [-14.34, 3.81, 1],
'J1643-1224' : [-12.85, 0.98, 0],
'J1824-2452A' : [-13.26, 5.02, 1],
'J1857+0943' : [-16.86, 7.49, 0],
'J1909-3744' : [-14.74, 4.05, 0], 
'J1939+2134' : [-14.33, 5.39, 2]
}

def read_ppta():
    # convert to P(f) at period of 1 year
    rvals = dict()
    for key in ppta:
        A = 10**ppta[key][0]
        g = ppta[key][1]
        P = A**2/(12*np.pi**2)*(365*86400)**2 # convert to s^2/yr
        rvals[key] = [P,ppta[key][1],ppta[key][2]]
    return rvals


nanog = read_nanograv()
ppta = read_ppta()
    
# this is the grid of spectral indices we ran the limits on
grid = np.arange(3,22)*(1./3)

# These are the "2" pulsars, i.e. those in common to the GWB sample and
# to either the published NG or PPTA results.  I have also checked the
# older EPTA sample but it doesn't contribute any new pulsars.
jnames = ['J0030+0451','J0613-0200','J1939+2134']

# The idea behind the "WN" ratio is to find a representative length of the
# data set where the TN predicted at that length is such that TN/WN is in
# rough agreement with what we get from the full TN analysis.  This way,
# we can do the same for pulsars we don't have TN limits for to get a rough
# idea of which ones might be possible to detect in the future.
for jname in jnames:
    print(jname)
    print('---------')
    lims = []
    for idx in grid:
        # this uses the non-TN marginalized limit -- appropriate for looking
        # for TN!!!
        jnames,wn,logl,lim1,lim2,gwb_dom,cods,cod_margs = load_unbinned_results_new(index=idx)
        mask = jnames == jname
        lims.append(lim1[mask])
    lims = np.ravel(lims)
    wn = wn[mask][0]*1e-12

    ip = interp1d(grid,np.log10(lims),kind='linear',fill_value='extrapolate')
    # convert from A to P(f) in s^2/yr
    if jname in nanog.keys():
        ipval = (10**ip(nanog[jname][1]))**2/(12*np.pi**2)*(86400*365)**-2
        y = nanog[jname][0]
        print('NANOGrav P(f) = %0.2g'%(y))
        print('   Fermi Lim. = %0.2g'%(ipval))
        print('        Ratio = %0.2f'%(y/ipval))
        print('    inv.Ratio = %0.2f'%(ipval/y))
        print('2x data Ratio = %0.2g'%(2**(1.0*nanog[jname][1])*y/ipval))
        print('""  inv.Ratio = %0.2g'%((2**(1.0*nanog[jname][1])*y/ipval)**-1))
        print('     WN Ratio = %0.2g'%(12**(1.0*nanog[jname][1])*y/wn))
    if jname in ppta.keys():
        ipval = (10**ip(ppta[jname][1]))**2/(12*np.pi**2)*(86400*365)**-2
        y = ppta[jname][0]
        print('PPTA     P(f) = %0.2g'%(y))
        print('   Fermi Lim. = %0.2g'%(ipval))
        print('        Ratio = %0.2f'%(y/ipval))
        print('    inv.Ratio = %0.2f'%(ipval/y))
        print('2x data Ratio = %0.2g'%(2**(1.0*ppta[jname][1])*y/ipval))
        print('""  inv.Ratio = %0.2g'%((2**(1.0*ppta[jname][1])*y/ipval)**-1))
        print('     WN Ratio = %0.2g'%(12**(1.0*ppta[jname][1])*y/wn))

jnames = ['J0437-4715','J1600-3053','J1713+0747','J1744-1134','J1747-4036','J2317+1439']
wn2 = np.loadtxt('ancillary_data/lat_msps.asc',usecols=[3]).transpose()
jnames2 = np.loadtxt('ancillary_data/lat_msps.asc',usecols=[0],dtype=str).transpose()
for jname in jnames:
    print(jname)
    print('---------')
    mask = jnames2 == jname
    wn = wn2[mask]
    if jname in nanog.keys():
        y = nanog[jname][0]
        print('NANO WN Ratio = %0.2g'%(12**(1.0*nanog[jname][1])*y/wn))
        print('NANO WN Ratioi = %0.2g'%((12**(1.0*nanog[jname][1])*y/wn)**-1))
        print('NANO WN2Ratio = %0.2g'%(24**(1.0*nanog[jname][1])*y/wn))
    if jname in ppta.keys():
        y = ppta[jname][0]
        print('PPTA WN Ratio = %0.2g'%(12**(1.0*ppta[jname][1])*y/wn))
        print('PPTA WN Ratioi = %0.2g'%((12**(1.0*ppta[jname][1])*y/wn)**-1))
        print('PPTA WN2Ratio = %0.2g'%(24**(1.0*ppta[jname][1])*y/wn))
