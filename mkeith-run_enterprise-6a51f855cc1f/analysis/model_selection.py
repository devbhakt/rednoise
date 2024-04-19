#!/usr/bin/env python

import argparse
import sys, os
import subprocess
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#import numpy as np
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
parser.add_argument('-w', '--width', type=int, default=0, help='Boxcar width (days), defualt: 6*cadence')
parser.add_argument('-s', '--step', type=int, default=0, help='Size of stride (days), defualt: 3*cadence')
parser.add_argument('-a', '--gap', type=float, default=0, help='Threshold of gap (days), defualt: 10*cadence')
parser.add_argument('-u', '--taug', type=int, default=[200]*100, nargs='+', help='Replace GLF1 with change of spin frequency tau_g days after the glitch respectively')
parser.add_argument('-g', '--glitches', type=int, default=[], nargs='+', help='Glitches that need to split tim file for')
parser.add_argument('-m', '--multiple', type=int, default=[], nargs='+', help='Multiple glitches that need to split tim file together for, the first entry is the start glitch, the middle entries is the large glitches(max: 2) to be fitted, the last entry is the end glitch')
parser.add_argument('-r', '--recoveries', type=int, default=[0, 1], nargs='+', help='Number of recoveries to fit in models for all large glitches')
parser.add_argument('--glf2', type=str, default="a", help='Include GLF2 term in glitch model')
parser.add_argument('--f2', '--glf2-range', type=float, default=10, help='The prior range for glf2')
parser.add_argument('--glf0d', '--glf0d-range', type=float, default=0.8, help='The prior range for glf0d')
parser.add_argument('--glep', '--glep-range', type=float, default=2, help='The prior range for glep')
parser.add_argument('--sigma', '--measured-sigma', type=float, default=[100, 100, 100, 100], nargs='+', help='Minus/Plus sigma range of GLF0(instant), and Minus/Plus sigma range of GLF0(T=taug) respectively')
parser.add_argument('--split', '--gltd-split', type=float, default=[2.0, 2.3], nargs='+', help='Where to split gltd priors (in log10) for double and triple recoveries respectively')
parser.add_argument('--small', '--small-glitches', action='store_false', help='Turn on tempo2 fitting for the small glitches in split par if they are included in split tim')
parser.add_argument('--pre', '--pre-glitch', action='store_false', help='Use the best model(if exists) of previous glitch in the split par')
#parser.add_argument('-g', '--glep', type=float, default=[], nargs='+', help='Glitch epochs(MJD)')
#parser.add_argument('-d', '--data', help='Stride data text file', required=True)
args = parser.parse_args()
    
# Set Pulsar class sfpsr, load info, generate truth file
if args.glf2 is "n":
    g2 = [False]
elif args.glf2 is "y":
    g2 = [True]
else:
    g2 = [False, True]
sfpsr = pg.Pulsar(args.parfile, args.timfile, glf0t=args.taug)
sfpsr.delete_null()
sfpsr.print_info()
#sfpsr.generate_truth()

# Convert double recovery to tempo2 style, chop tim file, update info
#sfpsr.tidy_glitch(chop=5000)
#sfpsr.load_info()
#sfpsr.delete_null()

# Generating tim files for individual glitches, par files for different models, and run MCMC fit to find best model
for gi in args.glitches:
    sfpsr.split_tim(glitchnum=gi)
    for exp in args.recoveries:
        for gf2 in g2:
            if gf2 is True and exp==3:
                continue
            sfpsr.split_par(glitchnum=gi, recoveries=exp, GLF2=gf2, small_glitches=args.small, pre_gli=args.pre)
            sfpsr.MCMC_fit(glitchnum=gi, recoveries=exp, GLF2=gf2, gleprange=args.glep, glf0drange=args.glf0d, sigma=args.sigma, gltdsplit=args.split, glf2range=args.f2)
        #sfpsr.post_par(glitchnum=gi)
    sfpsr.best_model(glitchnum=gi, recoveries=None, GLF2=args.glf2)

if len(args.multiple)>0:
    start = args.multiple[0]
    end = args.multiple[-1]
    gi = args.multiple[1:-1]
    if len(gi)==1:
        gi = gi[0]
    sfpsr.split_tim(glitchnum=str(gi[0]), startnum=start, endnum=end)
    for exp in args.recoveries:
        for gf2 in g2:
            if gf2 is True and exp==3:
                continue
            sfpsr.split_par(glitchnum=gi, recoveries=exp, GLF2=gf2, small_glitches=args.small, pre_gli=args.pre)
            sfpsr.MCMC_fit(glitchnum=gi, recoveries=exp, GLF2=gf2, gleprange=args.glep, glf0drange=args.glf0d, sigma=args.sigma, gltdsplit=args.split, glf2range=args.f2)
        #sfpsr.post_par(glitchnum=args.multiple[1])
    sfpsr.best_model(glitchnum=args.multiple[1], recoveries=None, GLF2=args.glf2)
