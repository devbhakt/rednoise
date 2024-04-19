#!/usr/bin/env python


import argparse



parser = argparse.ArgumentParser(description="Run 'cholspec' spectral analalysis code")
parser.add_argument('par')
parser.add_argument('tim')
parser.add_argument("--fmax", default=None, type=float, help="Max freq in per day (default median spacing)")
parser.add_argument("--fmax-limit", default=None, type=float, help="use median spacing up to this limit")
parser.add_argument("--t2model", default=None, type=str, help="Use this T2model file")
parser.add_argument("--refmodel", default=None, type=str, help="Overplot this reference model")
parser.add_argument("--reftn", default=None, nargs=2, type=float, help="Overplot reference TN model (A,gamma)")
parser.add_argument("--reftn-qp", default=None, nargs=6, type=float, help="Overplot reference TN model (A,gamma)")
parser.add_argument("--plot-t2", default=None, type=str, help="cholSpectra output")
parser.add_argument("--simmodel", default=None, type=str, help="Overplot this simulated model")
parser.add_argument("--export-sim", action='store_true')
parser.add_argument("--test-fit", action='store_true')
parser.add_argument("--savefig", default=None, type=str, help="Save Figure")
parser.add_argument("--savetxt", action='store_true')

parser.add_argument("--plot-samples",default=None,type=str,nargs="*")
parser.add_argument("--samples-percentiles",action='store_true')

args=parser.parse_args()

import sys

import matplotlib
if args.savefig:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from enterprise.pulsar import Pulsar

from spectrum import cholspec

from t2model import *
from scipy import linalg

def ppplot(f2,ff,color,alpha):
    if args.samples_percentiles:
        upper = np.percentile(ff, 97.5, axis=0)
        lower = np.percentile(ff, 2.5, axis=0)
        plt.loglog(f2, upper, color=color, alpha=0.8,ls=':')
        plt.loglog(f2, lower, color=color, alpha=0.8,ls=':')
    else:
        for fff in ff:
            plt.loglog(f2, fff, color=color, alpha=alpha)

psr=Pulsar(args.par,args.tim,drop_t2pulsar=False)
is_binary=False
logA=gamma=F0=F1=PB=0
with open(args.par) as f:
    for line in f:
        split_line = line.split()
        if len(split_line)==0:
            continue
        if split_line[0] == 'TNRedAmp':
            logA = float(split_line[1])
        elif split_line[0] == 'TNRedGam':
            gamma = float(split_line[1])
        elif split_line[0] == 'F0':
            F0 = float(split_line[1])
        elif split_line[0] == 'F1':
            F1 = float(split_line[1])
        elif split_line[0] == 'PB':
            is_binary=True
            PB=float(split_line[1])

psr.sort_data()
toa = psr.toas/86400.0 # convert back to days for sanity
ntoa=len(toa)

toa_spacing = np.diff(toa)
if args.fmax==None:
    args.fmax = 0.5/np.median(toa_spacing)
    if not args.fmax_limit is None:
        args.fmax=min(args.fmax_limit,args.fmax)
tspan = np.amax(toa)-np.amin(toa)
fc_yr = 365.25/tspan / 4.0
# compute CVM
print("Tspan={:.1f} days, using fc = {:.4g} per year".format(tspan,fc_yr))
print("Compute CVM")

orig_model_psd = None
if args.t2model:
    print("Using t2model parameters")
    red_cvm,t2funcs = read_t2_model(args.t2model,psr)
elif logA != 0 and gamma != 0:
    print("Using TN parameters")
    red_cvm = cholspec.getC(toa, logA, gamma, fc_yr)
else:
    red_cvm = np.zeros((ntoa,ntoa))



if args.test_fit:
    ## TESTTESTTEST
    f2,psd = np.loadtxt("t2.psd",usecols=(0,1),unpack=True)

    plt.loglog(f2,psd)
    if args.t2model:
        for fn in t2funcs:
            plt.loglog(f2, fn(f2), color='lightgreen', label='T2Model')
    plt.show()
    design_matrix = psr.t2pulsar.designmatrix()


    def qrfit(dm, b):
        """Least Squares fitting useing the QR decomposition"""
        q, r = linalg.qr(dm, mode='economic')
        p = np.dot(q.T, b)

        ri = linalg.inv(r)
        param = np.dot(linalg.inv(r), p)
        newcvm = ri.dot(ri.T)
        model = param.dot(dm.T)
        postfit = b - model
        chisq = np.sum(np.power(postfit, 2))
        newcvm *= (chisq / float(len(b) - len(param)))
        return param, newcvm, chisq

    CVM = red_cvm + np.diag(psr.toaerrs**2)
    L = linalg.cholesky(CVM, lower=True)

    log_det_L = np.sum(np.log(np.diag(L)))

    residuals = psr.residuals


    white_residuals = linalg.solve_triangular(L, residuals, lower=True)

    white_DM = linalg.solve_triangular(L, design_matrix, lower=True)

    param, p_cvm, chisq = qrfit(white_DM, white_residuals)
    pname=psr.fitpars
    for i in range(len(param)):
        print("{:02d} {:10s} {:12.9g} {:12.9g}".format(i,pname[i],param[i],np.sqrt(p_cvm[i][i])))
    print("CHISQ",chisq)


print("Compute spectrum fmax={:.4g} per day".format(args.fmax))
f, psd_f_yr3, complex_spec, loglike = cholspec.cholspec(psr.residuals,toa,psr.toaerrs,red_cvm,f_max=args.fmax)

f *= 365.25


fig=plt.figure()

f2 = np.logspace(np.log10(np.amin(f)),np.log10(np.amax(f)),1024)

df = 86400.0 / (np.amax(psr.toas) - np.amin(psr.toas))

if not args.plot_samples is None:
    for ff in args.plot_samples:
        with open(ff) as infile:
            line = infile.readline()
            pars = np.array(line.split())

        dat = np.loadtxt(ff,skiprows=1).T
        npar,nsamp = dat.shape
        print(dat.shape)
        alpha = max(1/nsamp,0.01)
        ff = np.zeros((nsamp, len(f2)))
        if "TN_QpF0" in pars:
            ## temponest mode
            amps=dat[pars=="TNRedAmp",:][0]
            gams=dat[pars=="TNRedGam",:][0]
            f0s=dat[pars=="TN_QpF0",:][0]
            rats=dat[pars=="TN_QpRatio",:][0]
            sigs=dat[pars=="TN_QpSig",:][0]
            lams=dat[pars=="TN_QpLam",:][0]
            for i,pp in enumerate(zip(amps,gams,rats,f0s,sigs,lams)):
                ff[i] = cholspec.PSD_QP(f2 / 365.25, fc_yr, *pp,1/tspan)
            ppplot(f2,ff,'magenta',alpha)

        elif "TNRedAmp" in pars:
            ## temponest mode
            amps=dat[pars=="TNRedAmp",:][0]
            gams=dat[pars=="TNRedGam",:][0]
            for i in range(len(amps)):
                ff[i] = cholspec.PSD(f2/365.25, fc_yr, amps[i],gams[i])
            ppplot(f2,ff,'pink',alpha)


        elif "T2Chol_QpF0" in pars:
            amps = dat[pars == "T2Chol_RedA", :][0]
            alphas = dat[pars == "T2Chol_RedAlpha", :][0]
            f0s = dat[pars == "T2Chol_QpF0", :][0]
            rats = dat[pars == "T2Chol_QpRatio", :][0]
            sigs = dat[pars == "T2Chol_QpSig", :][0]
            lams = dat[pars == "T2Chol_QpLam", :][0]
            for i,pp in enumerate(zip(amps,alphas,rats,f0s,sigs,lams)):
                ff[i] =chol_red_model.pl_plus_qp_nudot_cutoff(f2, *pp, fref=0.1, fc=0.01,df=df)
            ppplot(f2, ff, 'lightgreen', alpha)


        elif "T2Chol_RedA" in pars:
            amps = dat[pars == "T2Chol_RedA", :][0]
            alphas = dat[pars == "T2Chol_RedAlpha", :][0]
            for i,pp in enumerate(zip(amps, alphas)):
                ff[i] = chol_red_model.pl_red(f2, *pp, fref=0.1, fc=0.01)  ## oops all hardcoded
            ppplot(f2,ff,'green',alpha)

plt.loglog(f,psd_f_yr3,color='blue')

if args.savetxt:
    np.savetxt("spec.txt", [f, psd_f_yr3])

if not args.plot_t2 is None:
    x,y = np.loadtxt(args.plot_t2,usecols=(0,1),unpack=True)
    plt.loglog(x,y,color='black',label=args.plot_t2)

if logA!=0 and gamma!=0:
    tnModel_psd = cholspec.PSD(f2/365.25, fc_yr, logA, gamma)
    plt.loglog(f2,tnModel_psd,color='red',label='TN')
    if args.savetxt:
        np.savetxt("tnmodel.txt", [f2, tnModel_psd])

if args.t2model:
    for fn in t2funcs:
        plt.loglog(f2,fn(f2),color='lightgreen',label='T2Model')
        if args.savetxt:
            np.savetxt("t2fun.txt",[f2,fn(f2)])

if args.refmodel:
    _, reffuncs = read_t2_model(args.refmodel, psr, makecvm=False)
    for fn in reffuncs:
        plt.loglog(f2, fn(f2), color='green', label='Ref T2Model')
        if args.savetxt:
            np.savetxt("ref.txt",[f2,fn(f2)])

if args.simmodel:
    CVM, simfuncs = read_t2_model(args.simmodel, psr, makecvm=args.export_sim)
    for fn in simfuncs:
        plt.loglog(f2, fn(f2), color='black',ls='--', label='Simulation Input')
    if args.export_sim:
        CVM += np.diag(psr.toaerrs**2)
        L = linalg.cholesky(CVM, lower=True)
        sim_residuals = np.dot(L,np.random.normal(size=ntoa))
        np.savetxt("cholspec.sim.res",np.array([psr.toas/86400.0,sim_residuals]).T)


if not args.reftn is None:
    tnModel_psd = cholspec.PSD(f2/365.25, fc_yr, *args.reftn)
    plt.loglog(f2, tnModel_psd, color='pink', label='Ref TN')
    if args.savetxt:
        np.savetxt("tn.txt",[f2,tnModel_psd])

if not args.reftn_qp is None:
    tnModel_psd = cholspec.PSD_QP(f2/365.25, fc_yr, *args.reftn_qp,1/tspan)
    plt.loglog(f2, tnModel_psd, color='magenta', label='Ref TN QP')
    if args.savetxt:
        np.savetxt("tnqp.txt",[f2,tnModel_psd])



plt.ylim(np.amin(psd_f_yr3)*0.25,np.amax(psd_f_yr3)*4)

plt.legend()
plt.xlabel("Frequency (yr$^{-1}$)")
plt.ylabel("PSD (yr$^3$)")
plt.title(psr.name)
if args.savefig:
    plt.savefig(args.savefig)
else:
    plt.show()


