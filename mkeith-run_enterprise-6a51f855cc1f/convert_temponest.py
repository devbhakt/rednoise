#!/usr/bin/env python

from __future__ import division, print_function

import matplotlib
matplotlib.use('Agg')

import argparse

import subprocess



parser=argparse.ArgumentParser(description="Convert temponest output to similar to 'run_enterprise.py'")
parser.add_argument('indir')
parser.add_argument('par')
parser.add_argument('tim')
parser.add_argument('--plot-chain',action='store_true', help='Make a plot of the chains')
parser.add_argument('--white-corner',action='store_true', help='Make the efac/equad corner plots')
parser.add_argument('--all-corner',action='store_true', help='Make corner plots with all params')
parser.add_argument('--pm-ecliptic',action='store_true', help='Generate ecliptic coords from pmra/pmdec')


args=parser.parse_args()

print(vars(args))

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from scipy.stats import rv_continuous
import glob

import sys
from enterprise.pulsar import Pulsar


import corner

from astropy import coordinates as coord
from astropy import units

from matplotlib.backends.backend_pdf import PdfPages


indir=args.indir
par=args.par
tim=args.tim
psr=Pulsar(par,tim,drop_t2pulsar=False)

parlines=[]
with open(par) as f:
    for line in f:
        parlines.append(line)
        e=line.split()
        if len(e) > 1:
            if e[0]=="F0":
                f0=float(e[1])
            elif e[0]=="PEPOCH":
                pepoch=float(e[1])
            elif e[0]=="POSEPOCH":
                posepoch=float(e[1])
            elif e[0]=="PMRA":
                orig_pmra=float(e[1])
            elif e[0]=="PMDEC":
                orig_pmdec=float(e[1])
            elif e[0]=="RAJ":
                psr_ra=e[1]
            elif e[0]=="DECJ":
                psr_dec=e[1]



psr_coord = coord.SkyCoord(psr_ra,psr_dec,unit=(units.hourangle,units.degree))


parfile = glob.glob("{}/*.par".format(indir))[0]

scalfile = glob.glob("{}/*-T2scaling.txt".format(indir))[0]

pewfile = glob.glob("{}/*-post_equal_weights.dat".format(indir))[0]

pnamefile = glob.glob("{}/*-.paramnames".format(indir))[0]

t2_a, t2_b = np.loadtxt(scalfile,usecols=(-2,-1),unpack=True)
t2_l = np.loadtxt(scalfile,usecols=(0),unpack=True,dtype=np.str)

pn = np.loadtxt(pnamefile,usecols=(1),dtype=np.str)
print(pn)
psrn="unk"    
nC=100
efacs=[]
equads=[]
with open(parfile) as f:
    for line in f:
        e=line.split()
        if e[0] == "PSRJ":
            psrname=e[1]
        if e[0] == "TNRedC":
            nC=int(e[1])
        if e[0] == "TNEF":
            efacs.append("TNEF {} {}".format(e[1],e[2]))
        if e[0] == "TNEQ":
            equads.append("TNEQ {} {}".format(e[1],e[2]))
            
        # I think temponest does something wierd when there is just one flag
        # assume that if there is one flag it's jbdfb
        if e[0] == "TNGlobalEF":
            efacs.append("TNEF -be jbdfb")
        if e[0] == "TNGLobalEQ":
            equads.append("TNEQ -be jbdfb")

pn = list(np.loadtxt(pnamefile,usecols=(1),dtype=np.str))

post_eq = np.loadtxt(pewfile)
for i in range(len(pn)):
    
    if pn[i]=="RedAmp":
        pn[i]="TNRedAmp"
    if pn[i] == "RedSlope":
        pn[i] = "TNRedGam"
    if pn[i].startswith("EFAC"):
        j=int(pn[i][4:])
        pn[i]=efacs[j-1]
        post_eq[:,i] = np.power(10.0,post_eq[:,i])
    if pn[i].startswith("EQUAD"):
        j=int(pn[i][5:])
        pn[i]=equads[j-1]
    if pn[i] in t2_l:
        #print(pn[i],np.mean(post_eq[:,i]),np.std(post_eq[:,i]))
        j=np.argwhere(t2_l==pn[i])[0][0]
        
        #print(t2_a[4],t2_b[4])
        post_eq[:,i]*=t2_b[j]
        post_eq[:,i]+=t2_a[j]
        if pn[i] == "F2":
            post_eq[:,i]*=1e27 # This scaling is used in run_enterprise.py
        #print(pn[i],np.mean(post_eq[:,i]),np.std(post_eq[:,i]))


pars=np.array(pn)
# NOTE: Post-equal weights file is not a "chain", but this allows me to re-use the code I had before                         
chain = post_eq         
burn=0 # there is no burn-in for multinest

                               
imax = np.argmax(chain[:,len(pars)])

pmax = chain[imax,:len(pars)]

def convert_f2(v):
    return v*1e-27

def convert_dmA(v):
    v = 10**v
    v *=  (utils.const.DM_K * 1400**2 * 1e12)
    vv = v*v
    v = np.log10(np.sqrt(vv))
    return v


ul68 = np.percentile(chain[burn:,:len(pars)],84.1,axis=0)
ll68 = np.percentile(chain[burn:,:len(pars)],15.9,axis=0)
ul = np.percentile(chain[burn:,:len(pars)],97.5,axis=0)
ll = np.percentile(chain[burn:,:len(pars)],2.5,axis=0)
means = np.mean(chain[burn:,:len(pars)],axis=0)
stds = np.std(chain[burn:,:len(pars)],axis=0)
medians = np.median(chain[burn:,:len(pars)],axis=0)

post_derived = chain[burn:,:len(pars)]
pars_derived = np.array(pars).copy()

print("")
print("Max LogLikelihood",pmax)
print("")
outpar=par+".post"
resfname=par+".results"
with open(resfname,"w") as resf:
    with open(outpar,"w") as outf:
        s=("{:20s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}".format("#param"," max-like"," mean"," std"," 2.5%"," 15.9%"," 50%"," 84.1%"," 97.5%"))

        print(s)
        resf.write(s+"\n")
        for line in parlines:
            e=line.split()
            if e[0] in pars:
                outf.write("# "+line)
            else:
                outf.write(line)
        for p,v,u,l,mean,median,sig,u68,l68 in zip(pars,pmax,ul,ll,means,medians,stds,ul68,ll68):
            if p=="F2":
                p="F2"
                v = convert_f2(v)
                mean = convert_f2(mean)
                sig = convert_f2(sig)
                l = convert_f2(l)
                l68 = convert_f2(l68)
                median = convert_f2(median)
                u68 = convert_f2(u68)
                u = convert_f2(u)
           
            elif "pm_angle" in p:
                p="PM-angle"
            elif "pm_amp" in p:
                p="PM"
            elif "px_px" in p:
                p='PX'
            elif "pm_pmra" in p:
                p='PMRA'
            elif "pm_pmdec" in p:
                p='PMDEC'
            elif p=="DM_A":
                v = convert_dmA(v)
                mean = convert_dmA(mean)
                sig = convert_dmA(sig)
                l = convert_dmA(l)
                l68 = convert_dmA(l68)
                median = convert_dmA(median)
                u68 = convert_dmA(u68)
                u = convert_dmA(u)
                outf.write("TNDMC %d\n"%nC)
                p="TNDMAmp"
            elif p=="DM_gamma":
                p="TNDMGam"
            elif "TNRedGam" in p:
                p="TNRedGam"
            elif "TNRedAmp" in p:
                p="TNRedAmp"
                outf.write("TNRedC %d\n"%nC)
                outf.write("TNRedFLow -0.301\n")
            s = "{:20s} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g}  {:< 10g} {:< 10g} {:< 10g} ".format(p,v,mean,sig,l,l68,median,u68,u)
            resf.write(s+"\n")
            print(s)
            if p=="PM":
                pmamp=v
                continue
            if p=="PM-angle":
                pmangle=v
                pos=psr.pos
                #psrdec = np.arcsin(pos[2]/np.sqrt(np.sum(pos*pos)))
                pmra = pmamp * np.cos(pmangle) #/np.cos(psrdec)
                pmdec = pmamp * np.sin(pmangle)
                outf.write("PMRA %s\n"%(pmra))
                outf.write("PMDEC %s\n"%(pmdec))
                continue
            outf.write("%s %s\n"%(p,v))

    print("\nSaved par file to '%s'\n"%outpar)


    print("Derived parameters:")
    if "PMRA" in pars and "PMDEC" in pars:
        pos=psr.pos
        #psrdec = np.arcsin(pos[2]/np.sqrt(np.sum(pos*pos)))
        i=0
        for p in pars:
            if "PMRA" in p:
                pmra = chain[burn:,i]
            if "PMDEC" in p:
                pmdec = chain[burn:,i]
            i+=1
        #pmz = pmra * np.cos(psrdec)
        pmamp = np.sqrt(pmra*pmra + pmdec*pmdec)
        mean = np.mean(pmamp)
        median=np.median(pmamp)
        u = np.percentile(pmamp,97.5)
        l = np.percentile(pmamp,2.5)
        v = pmamp[imax-burn]
        sig=np.std(pmamp)
        u68 = np.percentile(pmamp,84.1)
        l68 = np.percentile(pmamp,15.9)
        p = "PM"
        s = "{:20s} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g}  {:< 10g} {:< 10g} {:< 10g} ".format(p,v,mean,sig,l,l68,median,u68,u)
        print(s)
        resf.write(s+"\n")
       
        post_derived = np.concatenate((post_derived,np.array([pmamp]).T),axis=1)
        pars_derived = np.concatenate((pars_derived,[p]))

        if args.pm_ecliptic:
            pm_ra_cosdec=pmra*(units.mas/units.yr)
            pm_dec = pmdec*(units.mas/units.yr)
            psr_pmcoord = coord.ICRS(np.repeat(psr_coord.ra,len(pm_dec)),\
                    np.repeat(psr_coord.dec,len(pm_dec)),\
                    pm_ra_cosdec=pm_ra_cosdec,\
                    pm_dec=pm_dec)
            ecliptic = psr_pmcoord.transform_to(coord.BarycentricTrueEcliptic)
            pm_elat = ecliptic.pm_lat.to((units.mas/units.yr)).value
            pm_elon = ecliptic.pm_lon_coslat.to((units.mas/units.yr)).value 

            mean = np.mean(pm_elat)
            median=np.median(pm_elat)
            u = np.percentile(pm_elat,97.5)
            l = np.percentile(pm_elat,2.5)
            v = pm_elat[imax-burn]
            sig=np.std(pm_elat)
            u68 = np.percentile(pm_elat,84.1)
            l68 = np.percentile(pm_elat,15.9)
            p = "PMELAT"
            s = "{:20s} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g}  {:< 10g} {:< 10g} {:< 10g} ".format(p,v,mean,sig,l,l68,median,u68,u)
            print(s)
            resf.write(s+"\n")
            post_derived = np.concatenate((post_derived,np.array([pm_elat]).T),axis=1)
            pars_derived = np.concatenate((pars_derived,[p]))

            mean = np.mean(pm_elon)
            median=np.median(pm_elon)
            u = np.percentile(pm_elon,97.5)
            l = np.percentile(pm_elon,2.5)
            v = pm_elon[imax-burn]
            sig=np.std(pm_elon)
            u68 = np.percentile(pm_elon,84.1)
            l68 = np.percentile(pm_elon,15.9)
            p = "PMELON"
            s = "{:20s} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g}  {:< 10g} {:< 10g} {:< 10g} ".format(p,v,mean,sig,l,l68,median,u68,u)
            print(s)
            resf.write(s+"\n")

            post_derived = np.concatenate((post_derived,np.array([pm_elon]).T),axis=1)
            pars_derived = np.concatenate((pars_derived,[p]))


# In[10]:

post = chain[burn:,:len(pars)]


if args.plot_chain:
    x=np.arange(chain.shape[0])
    with PdfPages("%s.chain.pdf"%psrname) as pdf:
        for i in range(len(pars)):
            fig=plt.figure(figsize=(16,8))
            plt.plot(x[:burn],chain[:burn,i],'.',color='gray')
            plt.plot(x[burn:],chain[burn:,i],'.',color='k')
            plt.title("%s"%pars[i])
            plt.ylabel("%s"%pars[i])
            pdf.savefig()
            plt.close()


m_f2 = (pars=='F2')
i_ef = np.array([i for i, v in enumerate(pars) if 'TNEF' in v],dtype=np.int)
i_eq = np.array([i for i, v in enumerate(pars) if 'TNEQ' in v],dtype=np.int)

m_ef = np.zeros(len(pars),dtype=np.bool)
m_ef[i_ef] = True
m_eq = np.zeros(len(pars),dtype=np.bool)
m_eq[i_eq] = True

n_w=np.logical_not(np.logical_or(m_ef,m_eq))
imax = np.argmax(chain[:,len(pars)])
pmax = chain[imax,:len(pars)]

fig=corner.corner(post[:,n_w], labels=pars[n_w], smooth=True,truths=pmax[n_w])

fig.savefig("%s.corner.pdf"%psrname)

nderived=len(pars_derived)-len(pars)
if nderived > 0:
    derived = np.pad(n_w,(0,nderived),'constant',constant_values=(True))
    fig=corner.corner(post_derived[:,derived], labels=pars_derived[derived], smooth=True)
    fig.savefig("%s.corner_derived.pdf"%psrname)




if args.all_corner:
    fig=corner.corner(post, labels=pars, smooth=True);
    fig.savefig("%s.corner_all.pdf"%psrname)



if args.white_corner:
    fig=corner.corner(post[:,m_ef], labels=pars[m_ef], smooth=True);
    fig.savefig("%s.corner_efac.pdf"%psrname)
    fig=corner.corner(post[:,m_eq], labels=pars[m_eq], smooth=True);
    fig.savefig("%s.corner_equad.pdf"%psrname)




subprocess.call(['tempo2','-f',outpar,tim,'-qrfit','-outpar',par+'.final'])
