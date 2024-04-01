#!/usr/bin/env python

import argparse
import sys, os
import subprocess
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
import numpy as np
#import numpy.ma as ma
import pulsar_glitch as pg
#from __future__ import print_function
#from scipy.optimize import curve_fit
#from scipy import linalg
#from astropy import coordinates as coord
#from astropy import units as u
#from matplotlib.ticker import FormatStrFormatter
#from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#from mpl_toolkits.axes_grid.inset_locator import inset_axes



parser = argparse.ArgumentParser(description='Model selection routine for fitting over glitches. Written by Y.Liu (yang.liu-50@postgrad.manchester.ac.uk).')
parser.add_argument('-p', '--parfile', type=str, help='Parameter file', required=True)
parser.add_argument('-t', '--timfile', type=str, help='TOA file', required=True)
#parser.add_argument('-w', '--width', type=int, default=0, help='Boxcar width (days), defualt: 6*cadence')
#parser.add_argument('-s', '--step', type=int, default=0, help='Size of stride (days), defualt: 3*cadence')
#parser.add_argument('-a', '--gap', type=float, default=0, help='Threshold of gap (days), defualt: 10*cadence')
parser.add_argument('-u', '--taug', type=int, default=[200]*100, nargs='+', help='Replace GLF1 with change of spin frequency tau_g days after the glitch respectively')
parser.add_argument('-g', '--glitches', type=int, default=[], nargs='+', help='Glitches that need to split tim file for')
#parser.add_argument('-m', '--multiple', type=list, default=[], nargs='+', help='Multiple glitches that need to split tim file together for')
#parser.add_argument('-r', '--recoveries', type=int, default=[0, 1], nargs='+', help='Number of recoveries to fit in models for all large glitches')
#parser.add_argument('--glf2', type=str, default="a", help='Include GLF2 term in glitch model')
#parser.add_argument('--f2', '--glf2-range', type=float, default=10, help='The prior range for glf2')
#parser.add_argument('--glf0d', '--glf0d-range', type=float, default=0.8, help='The prior range for glf0d')
#parser.add_argument('--glep', '--glep-range', type=float, default=2, help='The prior range for glep')
#parser.add_argument('--sigma', '--measured-sigma', type=float, default=[100, 100, 100, 100], nargs='+', help='Minus/Plus sigma range of GLF0(instant), and Minus/Plus sigma range of GLF0(T=taug) respectively')
#parser.add_argument('--split', '--gltd-split', type=float, default=[2.0, 2.3], nargs='+', help='Where to split gltd priors (in log10) for double and triple recoveries respectively')
#parser.add_argument('--small', '--small-glitches', action='store_false', help='Turn on tempo2 fitting for the small glitches in split par if they are included in split tim')
#parser.add_argument('--pre', '--pre-glitch', action='store_false', help='Use the best model(if exists) of previous glitch in the split par')
parser.add_argument('--nr', '--nrecoveries', type=int, default=[], nargs='+', help='Number of recoveries in the best model for each glitch')
parser.add_argument('--nglf2', type=int, default=[], nargs='+', help='Include GLF2 term for the best model of these glitches')
#parser.add_argument('-a', '--gap', type=float, default=0, help='Threshold of gap (days), defualt: 10*cadence')
#parser.add_argument('-g', '--glep', type=float, default=[], nargs='+', help='Glitch epochs(MJD)')
#parser.add_argument('-d', '--data', help='Stride data text file', required=True)
args = parser.parse_args()
    
# Set Pulsar class sfpsr, load info, generate truth file
sfpsr = pg.Pulsar(args.parfile, args.timfile, glf0t=args.taug)
sfpsr.delete_null()

sum_results = "sum_" + args.parfile.split("_")[1] + ".results"
sfpsr.noise_slt = sfpsr.read_results_file(sum_results, noise=True)
print("<<< The noise solution for pulsar {} is {} >>>".format(sfpsr.psrn, sfpsr.noise_slt))
sfpsr.read_final_par(largeglitch=args.glitches)

# Update info with par results files from best model of each glitch, and extract parameters to slt file
idx_recovery = 0
for gi in args.glitches:
    if len(args.nglf2)==0:
        f2 = "a"
    elif gi in args.nglf2:
        f2 = "y"
    else:
        f2 = "n"
    if len(args.nr) > idx_recovery:
        sfpsr.best_model(glitchnum=gi, recoveries=args.nr[idx_recovery], GLF2=f2)
    else:
        sfpsr.best_model(glitchnum=gi, recoveries=None, GLF2=f2)
    idx_recovery += 1

# Load data analytic model from asc files for pulsar plots
t1, yf2 = np.loadtxt('deltanu_{}.asc'.format(sfpsr.psrn), usecols=(0, 1), unpack=True)
t2, yd, yd_model, yd2 = np.loadtxt('nudot_{}.asc'.format(sfpsr.psrn), usecols=(0, 1, 2, 3), unpack=True)
# yd is nudot, yd_model is nudot_mod, yd2 is nudot_sum = nudot + nudot_mod
t3, ydd, ydd_model, ydd2 = np.loadtxt('nuddot_{}.asc'.format(sfpsr.psrn), usecols=(0, 1, 2, 3), unpack=True)
# ydd is nuddot, ydd_model is nuddot_mod, ydd2 is nuddot_sum = nuddot + nuddot_mod

# Load stride data from panels
p1_mjd, p1_nu, p1_err = np.loadtxt('panel1_{}.txt'.format(sfpsr.psrn), unpack=True) # in 1e-6
p2_mjd, p2_nu, p2_err = np.loadtxt('panel2_{}.txt'.format(sfpsr.psrn), unpack=True) # in 1e-6
p3_mjd, p3_nu, p3_err = np.loadtxt('panel3_{}.txt'.format(sfpsr.psrn), unpack=True) # in 1e-6
p4_mjd, p4_nudot, p4_err = np.loadtxt('panel4_{}.txt'.format(sfpsr.psrn), unpack=True) # in 1e-10
p5_mjd, p5_nudot, p5_err = np.loadtxt('panel5_{}.txt'.format(sfpsr.psrn), unpack=True) # in 1e-10
p6_mjd, p6_nuddot, p6_err = np.loadtxt('panel6_{}.txt'.format(sfpsr.psrn), unpack=True) # in 1e-20

# Calculating analytic model terms
x = pg.mjd2sec(t1, sfpsr.pepoch)
tf1, tf2, tdf2 = sfpsr.psr_taylor_terms(x)
tglf0, tglf1, tglf2, texp1, texp2, texp3, tdglf1, tdglf2, tdexp = sfpsr.glitch_terms(t1)

# Calculate analytic model for panels
a1 = sfpsr.mask_glep(t1, yf2)
a2 = a1 - tf2 - tglf0 - tglf1 - tglf2
a3 = a2 - texp1 - texp2 - texp3
a4 = sfpsr.mask_glep(t1, yd2) # in 1e-15
a5 = a4 - (sfpsr.F1 + tdf2 + tdglf1 + tdglf2 - tdexp)*1e15 # in 1e-15
a6 = sfpsr.mask_glep(t1, ydd2) # in 1e-20

# Calculate new pepoch and F0/F1/F2 used for plots
pre_x = sfpsr.pp_new_pepoch(t1, a1, a4, a6)

# Update results
sfpsr.extract_results()
