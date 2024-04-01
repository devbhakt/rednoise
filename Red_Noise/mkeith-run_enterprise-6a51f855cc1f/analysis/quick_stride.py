#!/usr/bin/env python

import argparse
import sys, os
import subprocess
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
import numpy as np
import numpy.ma as ma
import pulsar_glitch as pg
#from __future__ import print_function
#from scipy.optimize import curve_fit
#from scipy import linalg
from astropy import coordinates as coord
from astropy import units as u
#from matplotlib.ticker import FormatStrFormatter
#from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#from mpl_toolkits.axes_grid.inset_locator import inset_axes



parser = argparse.ArgumentParser(description='Stride routine for fitting over a pulsar. Written by Y.Liu (yang.liu-50@postgrad.manchester.ac.uk).')
parser.add_argument('-p', '--parfile', type=str, help='Parameter file', required=True)
parser.add_argument('-t', '--timfile', type=str, help='TOA file', required=True)
parser.add_argument('-w', '--width', type=int, default=0, help='Boxcar width (days), defualt: 6*cadence')
parser.add_argument('-s', '--step', type=int, default=0, help='Size of stride (days), defualt: 3*cadence')
parser.add_argument('-a', '--gap', type=float, default=0, help='Threshold of gap (days), defualt: 10*cadence')
parser.add_argument('-u', '--taug', type=int, default=[200]*100, nargs='+', help='Replace GLF1 with change of spin frequency tau_g days after the glitch respectively')
parser.add_argument('-g', '--glitches', type=int, default=[], nargs='+', help='Glitches that need to split tim file for')
parser.add_argument('-m', '--multiple', type=list, default=[], nargs='+', help='Multiple glitches that need to split tim file together for')
parser.add_argument('-r', '--recoveries', type=int, default=[], nargs='+', help='Number of recoveries in the best model for each glitch')
parser.add_argument('--glf2', type=int, default=[], nargs='+', help='Include GLF2 term for the best model of these glitches')
parser.add_argument('--full', '--full-plot', action='store_true', help='Plot full version of stride fit plots')
#parser.add_argument('-g', '--glep', type=float, default=[], nargs='+', help='Glitch epochs(MJD)')
#parser.add_argument('-d', '--data', help='Stride data text file', required=True)
args = parser.parse_args()
    
# Set Pulsar class sfpsr, load info, generate truth file
sfpsr = pg.Pulsar(args.parfile, args.timfile, glf0t=args.taug)
sfpsr.delete_null()
#sfpsr.generate_truth()

# Convert double recovery to tempo2 style, chop tim file, update info
#sfpsr.tidy_glitch(chop=5000)
#sfpsr.load_info()
#sfpsr.delete_null()

# Merge the best model of each glitch into the final par, and update par info
sum_results = "sum_" + args.parfile.split("_")[1] + ".results"
sfpsr.noise_slt = sfpsr.read_results_file(sum_results, noise=True)
print("<<< The noise solution for pulsar {} is {} >>>".format(sfpsr.psrn, sfpsr.noise_slt))
sfpsr.read_final_par(largeglitch=args.glitches)

# Calculate data for pulsar plots
#sfpsr.noglitch_par() ###
#sfpsr.pp_create_files() ###
rx, ry, re, y_dat, freqs, pwrs = sfpsr.pp_calculate_data()

# Load data analytic model from asc files for pulsar plots
rx2, ry2, re2 = np.loadtxt('out2_{}.res'.format(sfpsr.psrn), usecols=(0, 5, 6), unpack=True)
t, y = np.loadtxt('ifunc_{}.asc'.format(sfpsr.psrn), usecols=(1, 2), unpack=True)
t1, yf2 = np.loadtxt('deltanu_{}.asc'.format(sfpsr.psrn), usecols=(0, 1), unpack=True)
t2, yd, yd_model, yd2 = np.loadtxt('nudot_{}.asc'.format(sfpsr.psrn), usecols=(0, 1, 2, 3), unpack=True)
# yd is nudot, yd_model is nudot_mod, yd2 is nudot_sum = nudot + nudot_mod
t3, ydd, ydd_model, ydd2 = np.loadtxt('nuddot_{}.asc'.format(sfpsr.psrn), usecols=(0, 1, 2, 3), unpack=True)
# ydd is nuddot, ydd_model is nuddot_mod, ydd2 is nuddot_sum = nuddot + nuddot_mod

# Check Test
print('Shapes of data')
print(np.shape(t), np.shape(t1), np.shape(t2), np.shape(y), np.shape(yf2), np.shape(yd), np.shape(yd_model), np.shape(yd2))
print(np.shape(rx), np.shape(ry), np.shape(re), np.shape(rx2), np.shape(ry2), np.shape(re2), np.shape(y_dat))
print(np.shape(freqs), np.shape(pwrs))

# Make pulsar plots
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(311)
plt.errorbar(rx, ry2, yerr=re2, color='k', marker='.', ls='None', ms=3.0, alpha=0.7)
plt.title('PSR '+sfpsr.psrn)
plt.xlabel('MJD')
plt.ylabel('residual (s)')
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle='--', color='purple', alpha=0.7)
ax2 = ax.twinx()
ax2.set_ybound(np.array(ax.get_ybound())*sfpsr.F0)
ax2.set_ylabel('Residual (turns)')
ax = fig.add_subplot(312)
plt.plot(t, y, color='green')
plt.errorbar(rx, ry, yerr=re, color='k', marker='.', ls='None', ms=3.0, alpha=0.7)
plt.xlabel('MJD')
plt.ylabel('residual (s)')
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle='--', color='purple', alpha=0.7)
ax2 = ax.twinx()
ax2.set_ybound(np.array(ax.get_ybound())*sfpsr.F0)
ax2.set_ylabel('Residual (turns)')
ax = fig.add_subplot(313)
plt.plot(t,y-y,color='green')
plt.errorbar(rx, ry-y_dat, yerr=re, color='k', marker='.', ls='None', ms=3.0, alpha=0.7)
plt.xlabel('MJD')
plt.ylabel('Residual - Model (s)')
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle='--', color='purple', alpha=0.7)
ax2 = ax.twinx()
ax2.set_ybound(np.array(ax.get_ybound())*sfpsr.F0)
ax2.set_ylabel('Residual - Model (turns)')
plt.savefig('residuals_{}.pdf'.format(sfpsr.psrn))
plt.close()

plt.figure(figsize=(16,9))
plt.plot(t, yd, color='blue')
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle='--', color='purple', alpha=0.7)
plt.title('PSR '+sfpsr.psrn)
plt.xlabel('MJD')
plt.ylabel('$\\dot{\\nu}$ ($10^{-15}$ Hz$^2$)')
plt.savefig('nudot_{}.pdf'.format(sfpsr.psrn))
plt.close()

fig = plt.figure(figsize=(16,9))
fig.suptitle('PSR '+sfpsr.psrn)
ax = fig.add_subplot(321)
plt.errorbar(rx, ry2, yerr=re2, color='k', marker='.', ls='None', ms=3.0, alpha=0.7)
plt.ylabel('residual (s)')
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle='--', color='purple', alpha=0.7)
ax3 = ax.twiny()
ax3.set_xbound((np.array(ax.get_xbound())-53005)/365.25+2004) # MJD 53005 = 2004.01.01
ax3.xaxis.tick_top()
ax3.set_xlabel('Year')
ax3.xaxis.set_tick_params(direction='inout', labeltop=True)
ax.xaxis.set_tick_params(labelbottom=False, direction='in')
ax2 = ax.twinx()
ax2.set_ybound(np.array(ax.get_ybound())*sfpsr.F0)
ax2.set_ylabel('Residual (turns)')
ax = fig.add_subplot(323)
plt.plot(t, y, color='green')
plt.errorbar(rx, ry, yerr=re, color='k', marker='.', ls='None', ms=3.0, alpha=0.7)
plt.ylabel('residual (s)')
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle='--', color='purple', alpha=0.7)
ax.xaxis.set_tick_params(labelbottom=False, direction='in')
ax2 = ax.twinx()
ax2.set_ybound(np.array(ax.get_ybound())*sfpsr.F0)
ax2.set_ylabel('Residual (turns)')
ax = fig.add_subplot(325)
plt.plot(t, y-y, color='green')
plt.errorbar(rx, ry-y_dat, yerr=re, color='k', marker='.', ls='None', ms=3.0, alpha=0.7)
plt.xlabel('MJD')
plt.ylabel('Residual - Model (s)')
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle='--', color='purple', alpha=0.7)
ax.xaxis.set_tick_params(labelbottom=True, direction='inout')    
ax2 = ax.twinx()
ax2.set_ybound(np.array(ax.get_ybound())*sfpsr.F0)
ax2.set_ylabel('Residual - Model (turns)')
ax = fig.add_subplot(322)
plt.plot(t, 1e6*yf2, color='orange')
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle='--', color='purple', alpha=0.7)
plt.xlabel('MJD')
plt.ylabel('$\\Delta{\\nu}$ ($\mathrm{\mu}$Hz)')
ax.yaxis.set_label_position('right')
ax.yaxis.set_tick_params(labelleft=False, labelright=True, right=True, left=False, direction='in')
ax3 = ax.twiny()
ax3.set_xbound((np.array(ax.get_xbound())-53005)/365.25+2004)
ax3.xaxis.tick_top()
ax3.set_xlabel('Year')
ax = fig.add_subplot(324)
plt.plot(t, yd_model, color='lightblue', ls='--')
plt.plot(t, yd2, color='blue')
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle='--', color='purple', alpha=0.7)
plt.xlabel('MJD')
plt.ylabel('$\\dot{\\nu}$ ($10^{-15}$ Hz$^2$)')
ax.yaxis.set_label_position('right')
ax.yaxis.set_tick_params(labelleft=False, labelright=True, right=True, left=False, direction='in')
ax = fig.add_subplot(326)
plt.plot(t, yd, color='blue')
for ge in sfpsr.pglep:
    plt.axvline(ge, linestyle='--', color='purple', alpha=0.7)
plt.xlabel('MJD')
plt.ylabel('$\\Delta\\dot{\\nu}$ ($10^{-15}$ Hz$^2$)')
ax.yaxis.set_label_position('right')
ax.yaxis.set_tick_params(labelleft=False, labelright=True, right=True, left=False, direction='in')
plt.subplots_adjust(hspace=0, wspace=0.15)
plt.figtext(x=0.47, y=0.94, s='$P$:{:.0f} ms'.format(1000.0/sfpsr.F0), horizontalalignment='center')
plt.figtext(x=0.53, y=0.94, s='$\\dot{{P}}$:{:.0g}'.format(-sfpsr.F1/(sfpsr.F0**2)), horizontalalignment='center')
if sfpsr.PB == None:
    pass
elif sfpsr.PB > 0:
    plt.figtext(x=0.59, y=0.94, s='$P_B$:{:.1g}'.format(sfpsr.PB), horizontalalignment='center')
plt.savefig('combined_{}.pdf'.format(sfpsr.psrn))

plt.figure()
plt.figtext(x=0.47, y=0.94, s='$P$:{:.0f} ms'.format(1000.0/sfpsr.F0), horizontalalignment='center')
plt.figtext(x=0.53, y=0.94, s='$\\dot{{P}}$:{:.0g}'.format(-sfpsr.F1/(sfpsr.F0**2)), horizontalalignment='center')
if sfpsr.PB == None:
    pass
elif sfpsr.PB > 0:
    plt.figtext(x=0.59, y=0.94, s='$P_B$:{:.1g}'.format(sfpsr.PB), horizontalalignment='center')
plt.loglog(freqs, pwrs)
plt.title('PSR '+sfpsr.psrn)
plt.xlabel('Freq (yr^-1)')
plt.xlabel('Power (???)')
plt.savefig('pwrspec_{}.pdf'.format(sfpsr.psrn))
plt.close()

# Do stride fitting, calculate stride fitting results, save in panels
#sfpsr.sf_main(width=args.width, step=args.step, F1e=5e-15) ###
#sfpsr.sf_calculate_data() ###

# Load stride data from panels
p1_mjd, p1_nu, p1_err = np.loadtxt('panel1_{}.txt'.format(sfpsr.psrn), unpack=True) # in 1e-6
p2_mjd, p2_nu, p2_err = np.loadtxt('panel2_{}.txt'.format(sfpsr.psrn), unpack=True) # in 1e-6
p3_mjd, p3_nu, p3_err = np.loadtxt('panel3_{}.txt'.format(sfpsr.psrn), unpack=True) # in 1e-6
p4_mjd, p4_nudot, p4_err = np.loadtxt('panel4_{}.txt'.format(sfpsr.psrn), unpack=True) # in 1e-10
p5_mjd, p5_nudot, p5_err = np.loadtxt('panel5_{}.txt'.format(sfpsr.psrn), unpack=True) # in 1e-10
p6_mjd, p6_nuddot, p6_err = np.loadtxt('panel6_{}.txt'.format(sfpsr.psrn), unpack=True) # in 1e-20

# Mask ToA gap
t_mask, t_inverse = sfpsr.toa_gap(t1, gap=args.gap)
glep = sfpsr.pglep[0]
gleps = sfpsr.pglep

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
sf = pg.mjd2sec(p1_mjd, sfpsr.pepoch)
pre_sf = pg.mjd2sec(p1_mjd, sfpsr.pp_pepoch)

# Subtract F0 and F1 terms before the first glitch
ana1 = a1 + (sfpsr.F0 - sfpsr.pp_f0) + (sfpsr.F1*x - sfpsr.pp_f1*pre_x)
ana2 = ana1 - 0.5*sfpsr.F2*pre_x**2 - tglf0 - tglf1 - tglf2
ana3 = ana2 - texp1 - texp2 - texp3 
ana4 = a4  # in 1e-15
ana5 = a5 + 1e15*(sfpsr.F1 - sfpsr.pp_f1 + tdf2 - sfpsr.F2*pre_x) # in 1e-15
ana6 = a6 #- 1e20*sfpsr.F2 # in 1e-20

# Subtract F0 and F1 terms befroe the first glitch
stf1 = p1_nu + 1e6*((sfpsr.F0 - sfpsr.pp_f0) + (sfpsr.F1*sf - sfpsr.pp_f1*pre_sf)) # in 1e-6
stf2 = p2_nu + 1e6*((sfpsr.F0 - sfpsr.pp_f0) + (sfpsr.F1*sf - sfpsr.pp_f1*pre_sf) + 0.5*sfpsr.F2*(sf**2 - pre_sf**2)) # in 1e-6
stf3 = p3_nu + 1e6*((sfpsr.F0 - sfpsr.pp_f0) + (sfpsr.F1*sf - sfpsr.pp_f1*pre_sf) + 0.5*sfpsr.F2*(sf**2 - pre_sf**2)) # in 1e-6
stf4 = p4_nudot # in 1e-10
stf5 = p5_nudot + 1e10*(sfpsr.F1 - sfpsr.pp_f1 + sfpsr.F2*sf - sfpsr.F2*pre_sf) # in 1e-10
stf6 = p6_nuddot #- 1e20*sfpsr.F2 # in 1e-20

def plot_range(func, a=0.15, b=0.15):
    '''Adjust ranges of plots'''
    bottom = min(func)
    top = max(func)
    span = top - bottom
    bottom -= a*span
    top += b*span
    return bottom, top

# Plot the spin frequency and spin-down rate residuals panels
if args.full is False: # Only plot the 1st, 2nd, 4th panel of stride fit plot
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(12, 9.6), gridspec_kw={'height_ratios': [1, 5, 5, 5]}) 
    fig.subplots_adjust(hspace=0)
    plt.subplot(411)
    for toa in sfpsr.toaseries:
        ax[0].axvline(toa, color='b', linestyle='dashed', alpha=0.2, linewidth=0.5) #ymax=0.05, 
    for gls in gleps:
        ax[0].axvline(gls, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    ax[0].set_yticks([])
    plt.ylabel('ToA', fontsize=24, labelpad=15)
    plt.subplot(412)
    ax[1].plot(t_mask, 1e6*ana1, 'k-', zorder=2, label='Analytic model')
    ax[1].plot(t_inverse, 1e6*ana1, 'g-', zorder=2, label='Extrapolation', alpha=0.8)
    ax[1].errorbar(p1_mjd, stf1, yerr=p1_err, marker='.', color='r', ecolor='r', linestyle='None', alpha=1, zorder=1, label='Stirde fit')
    plt.ylim(plot_range(1e6*ana1)[0], plot_range(1e6*ana1)[1])
    plt.ylabel(r'$\delta \nu$ ($\mu$Hz)', fontsize=24, labelpad=5)
    plt.yticks(fontsize=24)
    for gls in gleps:
        ax[1].axvline(gls, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    #plt.legend(loc='upper left')
    plt.subplot(413)
    ax[2].plot(t_mask, 1e6*ana2, 'k-', zorder=2)
    ax[2].plot(t_inverse, 1e6*ana2, 'g-', zorder=2, alpha=0.8)
    ax[2].errorbar(p2_mjd, stf2, yerr=p2_err, marker='.', color='r', ecolor='r', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        ax[2].axvline(gls, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    plt.ylim(plot_range(1e6*ana2)[0], plot_range(1e6*ana2)[1])
    plt.ylabel(r'$\delta \nu-\nu_{gp}$ ($\mu$Hz)', fontsize=24, labelpad=5)
    plt.yticks(fontsize=24)
    plt.subplot(414)
    ax[3].plot(t_mask, 1e-3*ana4, 'k-', zorder=2)
    ax[3].plot(t_inverse, 1e-3*ana4, 'g-', zorder=2, alpha=0.8)
    ax[3].errorbar(p4_mjd, 1e2*stf4, yerr=1e2*p4_err, marker='.', color='m', ecolor='m', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        ax[3].axvline(gls, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    plt.ylim(plot_range(1e-3*ana4, 0.5, 0.5)[0], plot_range(1e-3*ana4, 0.5, 0.5)[1])
    plt.ylabel(r'$\dot{\nu}$ ($10^{-12}$Hz$^{2}$)', fontsize=24, labelpad=5)
    plt.xlabel('Modified Julian Date', fontsize=24)
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)
    plt.savefig('nu_nudot_gp_{}.pdf'.format(sfpsr.psrn), format='pdf', dpi=400, bbox_inches='tight')
    plt.show()

if 1>0: # Some option for plot all panels
    fig, ax = plt.subplots(7, 1, sharex=True, figsize=(12, 18.6), gridspec_kw={'height_ratios': [1, 5, 5, 5, 5, 5, 5]})
    fig.subplots_adjust(hspace=0)
    plt.subplot(711)
    for toa in sfpsr.toaseries:
        ax[0].axvline(toa, color='b', linestyle='dashed', alpha=0.2, linewidth=0.5) #ymax=0.05, 
    for gls in gleps:
        ax[0].axvline(gls, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    ax[0].set_yticks([])
    plt.ylabel('ToA', fontsize=24, labelpad=15)
    plt.subplot(712)
    ax[1].plot(t_mask, 1e6*ana1, 'k-', zorder=2, label='Analytic model')
    ax[1].plot(t_inverse, 1e6*ana1, 'g-', zorder=2, label='Extrapolation', alpha=0.8)
    ax[1].errorbar(p1_mjd, stf1, yerr=p1_err, marker='.', color='r', ecolor='r', linestyle='None', alpha=1, zorder=1, label='Stirde fit')
    plt.ylim(plot_range(1e6*ana1)[0], plot_range(1e6*ana1)[1])
    plt.ylabel(r'$\delta \nu$ ($\mu$Hz)', fontsize=24, labelpad=5)
    plt.yticks(fontsize=24)
    for gls in gleps:
        ax[1].axvline(gls, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    #plt.legend(loc='upper left')
    plt.subplot(713)
    ax[2].plot(t_mask, 1e6*ana2, 'k-', zorder=2)
    ax[2].plot(t_inverse, 1e6*ana2, 'g-', zorder=2, alpha=0.8)
    ax[2].errorbar(p2_mjd, stf2, yerr=p2_err, marker='.', color='r', ecolor='r', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        ax[2].axvline(gls, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    plt.ylim(plot_range(1e6*ana2)[0], plot_range(1e6*ana2)[1])
    plt.ylabel(r'$\delta \nu-\nu_{gp}$ ($\mu$Hz)', fontsize=24, labelpad=5)
    plt.yticks(fontsize=24)
    #If exists recovery
    plt.subplot(714)
    ax[3].plot(t_mask, 1e6*ana3, 'k-', zorder=2)
    ax[3].plot(t_inverse, 1e6*ana3, 'g-', zorder=2, alpha=0.8)
    ax[3].errorbar(p3_mjd, stf3, yerr=p3_err, marker='.', color='r', ecolor='r', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        ax[3].axvline(gls, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    plt.ylim(plot_range(1e6*ana3, 0.5, 0.5)[0], plot_range(1e6*ana3, 0.5, 0.5)[1])
    plt.ylabel(r'$\delta \nu-\nu_{g}$ ($\mu$Hz)', fontsize=24, labelpad=5)
    plt.yticks(fontsize=24)
    plt.subplot(715)
    ax[4].plot(t_mask, 1e-3*ana4, 'k-', zorder=2)
    ax[4].plot(t_inverse, 1e-3*ana4, 'g-', zorder=2, alpha=0.8)
    ax[4].errorbar(p4_mjd, 1e2*stf4, yerr=1e2*p4_err, marker='.', color='m', ecolor='m', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        ax[4].axvline(gls, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    plt.ylim(plot_range(1e-3*ana4, 0.5, 0.5)[0], plot_range(1e-3*ana4, 0.5, 0.5)[1])
    plt.ylabel(r'$\dot{\nu}$ ($10^{-12}$Hz$^{2}$)', fontsize=24, labelpad=5)
    plt.yticks(fontsize=24)
    plt.subplot(716)
    ax[5].plot(t_mask, ana5, 'k-', zorder=2)
    ax[5].plot(t_inverse, ana5, 'g-', zorder=2, alpha=0.8)
    ax[5].errorbar(p5_mjd, 1e5*stf5, yerr=1e5*p5_err, marker='.', color='m', ecolor='m', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        ax[5].axvline(gls, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    plt.ylim(plot_range(ana5, 0.5, 0.5)[0], plot_range(ana5, 0.5, 0.5)[1])
    plt.ylabel(r'$\delta \dot{\nu}-\dot{\nu_{g}}$ ($10^{-15}$Hz$^{2}$)', fontsize=24, labelpad=5)
    plt.yticks(fontsize=24)
    plt.subplot(717)
    ax[6].plot(t_mask, 10*ana6, 'k-', zorder=2)
    ax[6].plot(t_inverse, 10*ana6, 'g-', zorder=2, alpha=0.8)
    ax[6].errorbar(p6_mjd, 10*stf6, yerr=10*p6_err, marker='.', color='m', ecolor='m', linestyle='None', alpha=1, zorder=1, markersize=4)
    for gls in gleps:
        ax[6].axvline(gls, color='k', linestyle='dotted', alpha=0.5, linewidth=2)
    plt.ylim(plot_range(10*ana6, 0.5, 0.5)[0], plot_range(10*ana6, 0.5, 0.5)[1])
    plt.ylabel(r'$\ddot{\nu}$ ($10^{-21}$Hz$^{3}$)', fontsize=24, labelpad=5)
    plt.xlabel('Modified Julian Date', fontsize=24)
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)
    plt.savefig('full_nu_nudot_gp_{}.pdf'.format(sfpsr.psrn), format='pdf', dpi=400, bbox_inches='tight')
    plt.show()

# Update info with par results files from best model of each glitch, and extract parameters to slt file
idx_recovery = 0
for gi in args.glitches:
    if len(args.glf2)==0:
        f2 = "a"
    elif gi in args.glf2:
        f2 = "y"
    else:
        f2 = "n"
    if len(args.recoveries) > idx_recovery:
        sfpsr.best_model(glitchnum=gi, recoveries=args.recoveries[idx_recovery], GLF2=f2)
    else:
        sfpsr.best_model(glitchnum=gi, recoveries=None, GLF2=f2)
    idx_recovery += 1
sfpsr.extract_results()
