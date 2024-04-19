#!/usr/bin/env python

from __future__ import division, print_function

import matplotlib
matplotlib.use('Agg')

import argparse
import emcee
import os

from multiprocessing import Pool
#from schwimmbad import MPIPool

parser=argparse.ArgumentParser(description="Run 'enterprise' on a single pulsar")
parser.add_argument('par')
parser.add_argument('tim')
parser.add_argument('--f2', type=float, default=0, help='range of f2 to search')
parser.add_argument('-N','--nsample', type=float, default=1e6, help='number of samples in MCMC')
parser.add_argument('-D','--dm',action='store_true', help='Enable DM variation search')
parser.add_argument('--Adm-max',type=float,default=-12,help='Max log10A_DM')
parser.add_argument('--Adm-min',type=float,default=-18,help='Min log10A_DM')
parser.add_argument('--no-red-noise',dest='red',default=True,action='store_false', help='Disable Power Law Red Noise search')
parser.add_argument('--jbo','-j',action='store_true',help='Use -be flag for splitting backends')
parser.add_argument('--be-flag','-f',help='Use specified flag for splitting backends')
parser.add_argument('--Ared-max','-A',type=float,default=-12,help='Max log10A_Red')
parser.add_argument('--Ared-min',type=float,default=-18,help='Min log10A_Red')
parser.add_argument('--red-gamma-max',type=float,default=8,help='Max gamma red')
parser.add_argument('--red-gamma-min',type=float,default=0,help='Min gamma red')
parser.add_argument('--red-prior-log',action='store_true',help='Use uniform prior in log space for red noise amplitude')
parser.add_argument('--red-ncoeff',type=int,default=60,help='Number of red noise coefficients (nC)')
parser.add_argument('-n','--no-sample',dest='sample',default=True,action='store_false', help='Disable the actual sampling...')
parser.add_argument('--no-white',dest='white',default=True,action='store_false', help='Disable efac and equad')
parser.add_argument('--white-prior-log',action='store_true',help='Use uniform prior in log space for Equad')
parser.add_argument('--efac-max',type=float, default=5, help='Max for efac prior')
parser.add_argument('--efac-min',type=float, default=0.2, help='Min for efac prior')
parser.add_argument('--ngecorr',action='store_true', help='Add ECORR for the nanograv backends')
parser.add_argument('--white-corner',action='store_true', help='Make the efac/equad corner plots')
parser.add_argument('--all-corner',action='store_true', help='Make corner plots with all params')
parser.add_argument('--pm',action='store_true', help='Fit for PMRA+PMDEC')
parser.add_argument('--px',action='store_true', help='Fit for parallax')
parser.add_argument('--px-range',type=float,default=10, help='Max parallax to search')
parser.add_argument('--px-verbiest',action='store_true', help=argparse.SUPPRESS) #help='Use Verbiest PX prior to correct for L-K bias')
parser.add_argument('--s1400',type=float,default=None, help=argparse.SUPPRESS)  #help='S1400, used for Verbiest PX prior')
parser.add_argument('--pm-angle',action='store_true', help='Fit for PM + angle')
parser.add_argument('--pm-range',type=float,default=10,help='Search range for proper motion (deg/yr)')
parser.add_argument('--models','-M',nargs='+',help='Add a model to model selection stack. Use e.g. --model "-D --f2 --no-red-noise"')
parser.add_argument('--outdir','-o',type=str,help="Output directory for chains etc")
parser.add_argument('--plot-chain',action='store_true', help='Make a plot of the chains')
parser.add_argument('--tspan-mult',type=float,default=2,help='Multiplier for tspan')
parser.add_argument('--pm-ecliptic',action='store_true', help='Generate ecliptic coords from pmra/pmdec')

parser.add_argument('--quasiperiodic','-Q', action='store_true',help='fit quasiperiodic (QP) model')
parser.add_argument('--Aqp-max',type=float,default=1,help='Max log10A_QP')
parser.add_argument('--Aqp-min',type=float,default=-4,help='Min log10A_QP')
parser.add_argument('--qp-prior-log',action='store_true',help='Use uniform prior in log space for QP amplitude')
parser.add_argument('--glitch-recovery',action='store_true', help='fit for glitch recoveries')
parser.add_argument('--glitches',type=int, nargs='+', help='Select glitches to fit')

parser.add_argument('--qp-f0-max',type=float,default=10.0,help='Max QP f0')
parser.add_argument('--qp-f0-min',type=float,default=0.1,help='Min QP f0')

parser.add_argument('--qp-sigma-max',type=float,default=10.0,help='Max QP sigma')
parser.add_argument('--qp-sigma-min',type=float,default=0.01,help='Min QP sigma')

parser.add_argument('--emcee',action='store_true', help='Use emcee sampler')
parser.add_argument('--cont',action='store_true', help='Continue existing chain (emcee)')
parser.add_argument('--nthread','-t',type=int, default=1, help="number of threads (emcee)")
parser.add_argument('--nwalkers',type=int, default=0, help="number of walkers (emcee)")
parser.add_argument('--test-threads',action='store_true', help='Test threading options (emcee)')
args=parser.parse_args()

print(vars(args))

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from scipy.stats import rv_continuous


import sys

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
import enterprise.constants as const
from enterprise.signals import deterministic_signals

import enterprise_extensions
from enterprise_extensions import models, model_utils

import corner
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from astropy import coordinates as coord
from astropy import units as units

from matplotlib.backends.backend_pdf import PdfPages


from quasiperiodic import quasiperiodic, create_quasiperiodic_basisfunction, FourierBasisGP_QP

import ent_emcee

par = args.par
tim=args.tim
f2range=args.f2

print("Read pulsar data")
psr=Pulsar(par,tim,drop_t2pulsar=False)

orig_toas=psr.t2pulsar.toas()
issorted=np.all(orig_toas[:-1] <= orig_toas[1:])
#mask=orig_toas[:-1] <= orig_toas[1:]
#i=0
#for b in mask:
#    if not b:
#        print("{:.20f}".format(orig_toas[i-1]))
#        print("{:.20f}".format(orig_toas[i]))
#        print("{:.20f}".format(orig_toas[i+1]))
#        print("==")
#    i+=1


psr.earthssb = psr.t2pulsar.earth_ssb
psr.observatory_earth = psr.t2pulsar.observatory_earth

orig_pmra=0
orig_pmdec=0
posepoch=-1
parlines=[]
glitches={}
with open(par) as f:
    for line in f:
        parlines.append(line)
        e=line.strip().split()
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
            elif e[0].startswith("GLEP"):
                glitches[e[0][4:]] = {'EP':float(e[1])}
            elif e[0].startswith("GLF0D"):
                glitches[e[0][5:]]['F0D'] = float(e[1])
            elif e[0].startswith("GLTD"):
                glitches[e[0][4:]]['TD'] = float(e[1])



psr_coord = coord.SkyCoord(psr_ra,psr_dec,unit=(units.hourangle,units.degree))

if posepoch < 0:
    print("Assume POSEPOCH=PEPOCH")
    posepoch=pepoch

if args.px and "PX" in psr.fitpars:
    print("ERROR: Can't fit for PX in both MCMC and least-squares")
    sys.exit(1)
if args.pm and "PMRA" in psr.fitpars:
    print("ERROR: Can't fit for PMRA in both MCMC and least-squares")
    sys.exit(1)
if args.pm and "PMDEC" in psr.fitpars:
    print("ERROR: Can't fit for PMDEC in both MCMC and least-squares")
    sys.exit(1)

if args.glitch_recovery:
    for gl in glitches:
        if "GLF0D{}".format(gl) in psr.fitpars or "GLTD{}".format(gl) in psr.fitpars:
            print("ERROR: Can't fit for GLF0D/GLTD in both MCMC and least-squares")
            sys.exit(1)


if (args.pm or args.px) and not issorted:
    print("ERROR: Toas must be sorted or wierd things happen with paralax and proper motion")
    sys.exit(1)

print("Done.")


def jbbackends(flags):
    backend_flags=flags['be']
    flagvals = np.unique(backend_flags)
    return {flagval: backend_flags == flagval for flagval in flagvals}


def flagbackends(flags,beflagval):
    backend_flags=flags[beflagval]
    flagvals = np.unique(backend_flags)
    return {flagval: backend_flags == flagval for flagval in flagvals}


@signal_base.function
def mjk_glitch_recovery(toas, f0, glep, f0d, log10_td, old_f0d, old_td):
    td = np.power(10.0,log10_td)
    f0d *= 1e-6

    t= toas/86400.0 - glep

    expf=np.zeros_like(toas) + 1.0
    m = t>=0
    expf[m] = np.exp(-t[m]/td) + 1.0
    phs = - f0d * td * 86400.0 * (1.0-expf)

    if old_td > 0:
        expf=np.zeros_like(toas) + 1.0
        m = t>=0
        expf[m] = np.exp(-t[m]/old_td) + 1.0
        phs += old_f0d * old_td * 86400.0 * (1.0-expf)

    return phs / f0



@signal_base.function
def mjkf2(toas,pepoch,f0,valf2):
    x = (toas-pepoch*86400.0)
    return -1e-27*valf2*x*x*x/6.0/f0

@signal_base.function
def mjkpm_angle(toas,pos,earthssb,posepoch,amp,angle,orig_pmra=0,orig_pmdec=0):
    #psrdec = np.arcsin(pos[2]/np.sqrt(np.sum(pos*pos)))

    pmra = amp * np.cos(angle) #/np.cos(psrdec)
    pmdec = amp * np.sin(angle)
    return mjkpm(toas,pos,earthssb,posepoch,pmra,pmdec,orig_pmra,orig_pmdec)
 


@signal_base.function
def mjkpx(toas,pos,earthssb,observatory_earth,px):
    # This is just the linearised parallax function from tempo2...
    pxconv=1.74532925199432958E-2/3600.0e3
    AULTSC=499.00478364

    rca = earthssb[:,0:3] + observatory_earth[:,0:3]
    rr = np.sum(rca*rca,axis=1)

    rcos1 = np.sum(pos*rca,axis=1)

    return px * 0.5*pxconv * (rr-rcos1*rcos1)/AULTSC

"""
class VerbiestParallaxPriorRV(rv_continuous):
    def _rvs(self):
        return self.i_invCDF(np.random.uniform(0,1,self._size))
    def _pdf(self,x):
        if x < self.a or x > self.b:
            return 0
        else:
            return self.i_PDF(x)
    def _logpdf(self,x):
        if x < self.a or x > self.b:
            return -np.inf
        else:
            return np.log(self.i_PDF(x))
    def _cdf(self,x):
        if x < self.a:
            return 0
        elif x > self.b:
            return 1
        else:
            return self.i_CDF(x)

def VerbiestParallaxPrior(gl,gb,PXmin=1e-3,PXmax=15,s1400=None,R0=8.5,EE=0.33,size=None):
    GalCo=np.array([gl,gb])
    EE=0.5
    R0=8.5
    PP=np.linspace(PXmin,PXmax,10000)
    RR = np.sqrt( R0 * R0 + np.power( np.cos( GalCo[0] ) / PP, 2.0 )
                           - 2.0 * R0 * np.cos( GalCo[0] ) * np.cos( GalCo[1] ) / PP );
    Pvol = np.power( RR, 1.9 ) \
               * np.exp( - abs( np.sin( GalCo[0] ) / PP ) / EE \
                  - 5.0 * ( RR - R0 ) / R0 ) * np.power( PP, -4.0 )

    Pvol /= np.max(Pvol)
    zz = Pvol

    if s1400 != None:
        Plum = np.power( PP, -1.0 ) *\
            np.exp( -0.5 * np.power( ( np.log10( s1400 ) - 2.0 * np.log10( PP ) + 1.1 )\
                / 0.9, 2.0 ) );
        Plum/=np.max(Plum)
        zz *= Plum


    PDF = zz/np.sum(zz)
    CDF = integrate.cumtrapz(PDF,initial=0)
    CDF -= np.amin(CDF)
    CDF /= np.amax(CDF)

    RV = VerbiestParallaxPriorRV(a=PXmin,b=PXmax)
    RV.i_PDF    = interpolate.interp1d(PP,PDF)
    RV.i_CDF    = interpolate.interp1d(PP,CDF)
    RV.i_invCDF = interpolate.interp1d(CDF, PP)

    class VerbiestParallaxPrior(parameter.Parameter):
        _prior=prior.Prior(RV)
        _size=size
        _gl=gl
        _gb=gb
        def __repr__(self):
            return '"{}":VerbiestParallax({},{})'.format(self.name, gl,gb) \
                + ('' if self._size is None else '[{}]'.format(self._size))
    return VerbiestParallaxPrior
"""

@signal_base.function
def mjkpm(toas,pos,earthssb,posepoch,pmra,pmdec,orig_pmra=0,orig_pmdec=0):
    # This is just the linearised proper motion function from tempo2...
    AULTSC=499.00478364
    rce = earthssb[:,0:3]
    re = np.sqrt(np.sum(rce*rce,axis=1))
   
    psrra=np.arctan2(pos[1],pos[0])+2*np.pi
    psrdec = np.arcsin(pos[2]/np.sqrt(np.sum(pos*pos)))
    
    axy=rce[:,2]/AULTSC
    s = axy/(re/AULTSC)
    deltae = np.arctan2(s,np.sqrt(1.0-s*s))
    alphae = np.arctan2(rce[:,1],rce[:,0])
  
    # Convert radians/second to degrees per year
    # 60.0*60.0*1000.0*86400.0*365.25/24.0/3600.0
    rs2dy = (180.0/np.pi)*1314900000

    # This will be pmra in rad/s
    v_pmra  = (pmra-orig_pmra) / rs2dy / np.cos(psrdec)
    v_pmdec = (pmdec-orig_pmdec) / rs2dy
    
    t0 = (toas/86400.0-posepoch)
    d_pmra = re * np.cos(deltae)*np.cos(psrdec)*np.sin(psrra-alphae) * t0
    d_pmdec = re*(np.cos(deltae)*np.sin(psrdec)*np.cos(psrra-alphae) - np.sin(deltae)*np.cos(psrdec))*t0
    
    return v_pmra*d_pmra + v_pmdec*d_pmdec

@signal_base.function
def powerlaw_nogw(f, log10_A=-16, gamma=5):
    df = np.diff(np.concatenate((np.array([0]), f[::2])))
    return ((10**log10_A)**2 *
            const.fyr**(gamma-3) * f**(-gamma) * np.repeat(df, 2))



def FourierBasisGP_DM(spectrum, components=20,
                   selection=Selection(selections.no_selection),
                   Tspan=None, name=''):
    """Convenience function to return a BasisGP class with a
    fourier basis."""

    basis = utils.createfourierdesignmatrix_dm(nmodes=components, Tspan=Tspan)
    BaseClass = gp_signals.BasisGP(spectrum, basis, selection=selection, name=name)

    class FourierBasisGP_DM(BaseClass):
        signal_type = 'basis'
        signal_name = 'dm noise'
        signal_id = 'dm_noise_' + name if name else 'dm_noise'

    return FourierBasisGP_DM




def makeModel(args):
    print("Set up model...")

    f2range=args.f2
    selection = selections.Selection(selections.by_backend)
    selflag='-f'
    if args.jbo:
        selection = selections.Selection(jbbackends)
        selflag='-be'
    if args.be_flag:
        selflag="-"+args.be_flag
        selection = selections.Selection(lambda flags: flagbackends(flags,args.be_flag))


    s = selection(psr)



    tm = gp_signals.TimingModel()
    model =  tm

    if args.white:
        medianlogerr=np.log10(np.median(psr.toaerrs))
        efac = parameter.Uniform(args.efac_min,args.efac_max)
        if(args.white_prior_log):
            equad = parameter.Uniform(medianlogerr-2,medianlogerr+2)
        else:
            equad = parameter.LinearExp(medianlogerr-2,medianlogerr+2)
        ef = white_signals.MeasurementNoise(efac=efac,selection=selection)
        eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
        model += ef
        model += eq
    else:
        ef = white_signals.MeasurementNoise(efac=parameter.Constant(1.0),selection=selection)
        model += ef

    if args.ngecorr:
        ngselection = selections.Selection(selections.nanograv_backends)
        ecorr = parameter.LinearExp(-9,-5)
        ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=ngselection)
        model += ec

    if args.red:
        # red noise (powerlaw with 30 frequencies)
        if args.red_prior_log:
            log10_A = parameter.Uniform(args.Ared_min,args.Ared_max)
        else:
            log10_A = parameter.LinearExp(args.Ared_min,args.Ared_max)
        gamma = parameter.Uniform(args.red_gamma_min,args.red_gamma_max)
        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        rn = gp_signals.FourierBasisGP(spectrum=pl, components=nC, Tspan=Tspan)
        model+=rn

    if args.quasiperiodic:
        # Quasiperiodic noise with a fourier basis
        if args.qp_prior_log:
            log10_A = parameter.Uniform(args.Aqp_min, args.Aqp_max)('QP_A')
        else:
            log10_A = parameter.LinearExp(args.Aqp_min, args.Aqp_max)('QP_A')
        qp_f0 = parameter.Uniform(args.qp_f0_min, args.qp_f0_max)('QP_f0')
        qp_sigma = parameter.Uniform(args.qp_sigma_min, args.qp_sigma_max)('QP_sigma')

        qp_law = quasiperiodic(log10_A=log10_A, f0=qp_f0, sigma=qp_sigma)

        qp_basis = create_quasiperiodic_basisfunction(sigma=qp_sigma, f0=qp_f0, Tspan=Tspan)

        qp = FourierBasisGP_QP(basis=qp_basis,spectrum=qp_law)

        model += qp

    if args.dm:
        log10_Adm = parameter.LinearExp(args.Adm_min,args.Adm_max)('DM_A')
        gamma_dm = parameter.Uniform(0,6)('DM_gamma')
        pldm = powerlaw_nogw(log10_A=log10_Adm, gamma=gamma_dm)
        dm = FourierBasisGP_DM(spectrum=pldm, components=nC, Tspan=Tspan)
        model+=dm

    if args.pm:
        pmrange=args.pm_range
        pmra=parameter.Uniform(-pmrange+orig_pmra,pmrange+orig_pmra)
        pmdec=parameter.Uniform(-pmrange+orig_pmdec,pmrange+orig_pmdec)
        pm = deterministic_signals.Deterministic(mjkpm(posepoch=posepoch,pmra=pmra,pmdec=pmdec,orig_pmra=orig_pmra,orig_pmdec=orig_pmdec),name='pm')
        model += pm

    if args.px:
        if args.px_verbiest:
            print("\n\nWarning: Using Verbiest PX prior (Not tested!!)\n\n")
            print("\n\nERROR: FAIL\n\n")
            sys.exit(1)
            """
            if f0 > 100:
                px_ee=0.5
            else:
                px_ee=0.33
            ra=np.arctan2(psr.pos[1],psr.pos[0])*u.radian
            dec=np.arcsin(psr.pos[2]/np.sqrt(np.sum(psr.pos*psr.pos)))*u.radian
            c = coord.SkyCoord(ra,dec)
            gal = c.transform_to(coord.Galactic)
            gl=gal.l.deg
            gb=gal.b.deg
            pxp=VerbiestParallaxPrior(gl,gb,s1400=args.s1400,EE=px_ee)
            """
        else:
            pxp=parameter.Uniform(0,args.px_range)
        px = deterministic_signals.Deterministic(mjkpx(px=pxp),name='px')
        model += px

    if args.pm_angle:
        pmrange=args.pm_range
        pmamp=parameter.Uniform(0,pmrange)
        pmangle=parameter.Uniform(0,2*np.pi)
        pm = deterministic_signals.Deterministic(mjkpm_angle(posepoch=posepoch,amp=pmamp,angle=pmangle),name='pm')
        model += pm

    if args.glitch_recovery:
        for gl in glitches:
            igl=int(gl[1:])
            if args.glitches==None or igl in args.glitches:
                f0d = parameter.Uniform(-1*f0,1*f0)("GLF0D{}".format(gl)) # units of 10^-6
                td = parameter.Uniform(0,4)("GLTD{}".format(gl))
                if "TD" in glitches[gl]:
                    old_td = parameter.Constant(glitches[gl]['TD'])
                else:
                    old_td = parameter.Constant(0)

                if "F0D" in glitches[gl]:
                    old_f0d = parameter.Constant(glitches[gl]['F0D'])
                else:
                    old_f0d = parameter.Constant(0)

                glexp = mjk_glitch_recovery(f0=parameter.Constant(f0), glep=parameter.Constant(glitches[gl]['EP']), f0d=f0d, log10_td=td, old_td=old_td, old_f0d=old_f0d)
                glexp_f = deterministic_signals.Deterministic(glexp,name="gl{}".format(gl))
                model += glexp_f


    if f2range > 0:
        f2 = mjkf2(pepoch=parameter.Constant(pepoch),f0=parameter.Constant(f0), valf2=parameter.Uniform(-f2range/1e-27,f2range/1e-27)('F2'))
        f2f = deterministic_signals.Deterministic(f2,name='f2')
        model+= f2f
    return model


nC=args.red_ncoeff
Tspan = psr.toas.max()-psr.toas.min()
print("Tspan=%.1f"%(Tspan/86400.))

Tspan*=args.tspan_mult
nC*=int(args.tspan_mult)


#print(psr.fitpars)


selflag='-f'
if args.jbo:
    selflag='-be'
if args.be_flag:
    selflag="-"+args.be_flag


if args.outdir == None:
    outdir='chains/'+psr.name+"/"
else:
    outdir=args.outdir

print("Output directory =",outdir)


if args.sample:
    if args.models==None:
        model = makeModel(args)
        pta = signal_base.PTA(model(psr))
        x0 = np.hstack([p.sample() for p in pta.params])
        sampler=model_utils.setup_sampler(pta,outdir)
        if args.emcee:
            nwalkers = args.nwalkers
            if nwalkers < 2*len(x0):
                nwalkers = 2*len(x0)

            if args.cont:
                scls,offs = np.loadtxt(os.path.join(outdir,"scloff"),unpack=True)
            else:
                vv = [np.hstack([p.sample() for p in pta.params]) for i in range(100)]
                scls = np.std(vv,axis=0)
                offs = np.mean(vv,axis=0)
                np.savetxt(os.path.join(outdir,"scloff"),np.array([scls,offs]).T)

            if args.test_threads:
                print("Testing threading... nwalkers=",nwalkers)
                p0=np.zeros((nwalkers,len(x0)))
                for i in range(nwalkers):
                    p0[i] = np.hstack([p.sample() for p in pta.params])
                    p0[i] -= offs
                    p0[i] /= scls
                import time
                for t in (range(args.nthread+1))[::2]:
                    if t==0:
                        t=1
                    with Pool(t,initializer = ent_emcee.init,initargs=[pta,offs,scls]) as pool:
                        start = time.time()
                        n=4
                        for j in range(n):
                            res = pool.map(ent_emcee.log_prob,(p0[i] for i in range(nwalkers)))
                        end = time.time()
                        delta_time = end - start
                        if t==1:
                            tone=delta_time
                        print("t={} {:.1f}s {:.1f}ms speedup={:.1f} {:.0f}%".format(t,delta_time/n, t*1000.*delta_time/n/len(p0),tone/delta_time, 100.*tone/delta_time/t))
                sys.exit(1)

            
            tpool = Pool(args.nthread,initializer = ent_emcee.init,initargs=[pta,offs,scls])

            #tpool = MPIPool()
            #ent_emcee.init(pta)
            #if not tpool.is_master():
            #    tpool.wait()
            #    sys.exit(0)

            filename = os.path.join(outdir,"chain.h5")
            backend = emcee.backends.HDFBackend(filename)
            if args.cont:
                print("CONTINUE EXISTING CHAIN!")
            else:
                backend.reset(nwalkers,len(x0))
            #moves=[(emcee.moves.StretchMove(),0.4),(emcee.moves.DEMove(), 0.5), (emcee.moves.DESnookerMove(), 0.1),]
            moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
            sampler = emcee.EnsembleSampler(nwalkers,len(x0),ent_emcee.log_prob,backend=backend,pool=tpool,moves=moves)
            p0=np.zeros((nwalkers,len(x0)))
            for i in range(nwalkers):
                p0[i] = np.hstack([p.sample() for p in pta.params])
                p0[i] -= offs
                p0[i] /= scls
    else:
        nmodels = len(args.models)
        mod_index = np.arange(nmodels)
        pta = dict.fromkeys(mod_index)
        i=0
        for m in args.models:
            margs=[args.par,args.tim]
            margs.extend(m.split())
            if args.jbo:
                margs.append("-j")

            if args.be_flag:
                margs.append("--be-flag")
                margs.append(args.be_flag)
            if args.red_prior_log:
                margs.append("--red-prior-log")

            margs.append("--efac-max")
            margs.append("{}".format(args.efac_max))
            margs.append("--efac-min")
            margs.append("{}".format(args.efac_min))

            pargs=parser.parse_args(margs)
            model=makeModel(pargs)
            pta[i] = signal_base.PTA(model(psr))
            i += 1
        super_model = model_utils.HyperModel(pta)
        x0 = super_model.initial_sample()
        sampler = super_model.setup_sampler(resume=False, outdir=outdir)
        if args.emcee:
            def log_prob(*args,**kwargs):
                return super_model.get_lnlikelihood(*args,**kwargs)+super_model.get_lnprior(*args,**kwargs)
            filename = os.path.join(outdir,"chain.h5")
            backend = emcee.backends.HDFBackend(filename)
            backend.reset(2*len(x0),len(x0))
            sampler = emcee.EnsembleSampler(2*len(x0),len(x0),log_prob,backend=backend)
            p0=np.zeros((2*len(x0),len(x0)))
            for i in range(2*len(x0)):
                p0[i] = super_model.initial_sample()

print("PSR Name: ",psr.name)
print("F0: ",f0)
print("PEPOCH: ",pepoch)
print("POSPOCH: ",posepoch)
print("")
print("'least-squares' fit parameters:")
i=1
for p in psr.fitpars:
    print("  {:2d}   {}".format(i,p))
    i+=1


pars = np.loadtxt(outdir + '/pars.txt', dtype=np.unicode_)
print("MCMC fit parameters:")
i=1
for p in pars:
    z=""
    if not args.models == None:
        for ii in range(len(args.models)):
            if p in pta[ii].param_names:
                z += "%d"%ii
            else:
                z += "_"


    print("  {:2d}   {:40s} {}".format(i,p,z))
    i+=1


# sampler for N steps
N = int(args.nsample)
if args.sample:
    if args.emcee:

        """
        state=p0
        for state in sampler.sample(p0, N, progress=True):
            print("T",state)

        """

        print("Launch EMCEE\n\n")

        if args.cont:
            p0=None
        state = sampler.run_mcmc(p0,1)
        for state in sampler.sample(state,iterations=N,progress=True):
            if sampler.iteration % 10:
                continue
            i = np.argmin(sampler.acceptance_fraction)
            state.log_prob[i] -= 1e99

            if sampler.iteration % 100:
                continue
            print(sampler.iteration)
            print("Acceptance rate: {:.2f} {:.2f} {}".format(np.mean(sampler.acceptance_fraction),np.amin(sampler.acceptance_fraction),i))

        """
        if args.cont:
            p0=None
        else:
            print("Burn period\n\n")
            state = sampler.run_mcmc(p0,N//10,progress=True)
            print("Mean Acceptance rate: ",np.mean(sampler.acceptance_fraction))
            print("Min Acceptance rate:  ",np.amin(sampler.acceptance_fraction))
            state.log_prob -= 1e99
        state = sampler.run_mcmc(state,N,progress=True)
        print("Mean Acceptance rate: ",np.mean(sampler.acceptance_fraction))
        print("Min Acceptance rate:  ",np.amin(sampler.acceptance_fraction))
        """
        # todo... option to terminate early
        tpool.close()
    else:
        sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)


# In[8]:


if args.emcee:
    filename = os.path.join(outdir,"chain.h5")
    scls,offs = np.loadtxt(os.path.join(outdir,"scloff"),unpack=True)
    reader = emcee.backends.HDFBackend(filename)

    tau = reader.get_autocorr_time(tol=0)
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    print("tau = {} burn = {} thin = {}".format(tau, burnin, thin))
    samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
    samples *= scls
    samples += offs
    log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)

    log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)
    pars = np.loadtxt(outdir + '/pars.txt', dtype=np.unicode_)

    N=len(samples)
    chain = np.concatenate( (samples, log_prob_samples[:, None], log_prior_samples[:, None],np.zeros(N)[:,None],np.zeros(N)[:,None]), axis=1)
    burn=int(0.25*chain.shape[0])

else:
    chain = np.loadtxt(outdir+'/chain_1.txt')
    pars = np.loadtxt(outdir + '/pars.txt', dtype=np.unicode_)
    burn = int(0.25 * chain.shape[0])


if not args.models == None:
    msfname=par+".models"
    with open(msfname,"w") as msf:
        # We want to find the best model...
        odd_vals = np.zeros(len(args.models))
        samples = chain[burn:,-5]
        for i in range(len(args.models)):
            mask = np.rint(samples) == i
            odd_vals[i] = float(np.sum(mask))
        bestmodel = np.argmax(odd_vals)
        print("Doing model selection...")
        print("Best Model: {} '{}'".format(bestmodel,args.models[bestmodel]))
        msf.write("Best Model: {} '{}'\n".format(bestmodel,args.models[bestmodel]))
        print("Odds ratio:")
        for i in range(len(args.models)):
            if odd_vals[i] == 0:
                frac = 1.0 / odd_vals[bestmodel]
            else:
                frac = odd_vals[i]/odd_vals[bestmodel]
            
            print("Model P(%d)/P(%d): %.6lf   %.1f"%(i,bestmodel,frac,np.log(frac)))
            msf.write("Model P(%d)/P(%d): %.6lf   %.1f\n"%(i,bestmodel,frac,np.log(frac)))
        #print(model_utils.odds_ratio(chain[burn:,-5],models=[0,1]))

        post = chain[burn:,:len(pars)]

        m_f2 = (pars=='F2')
        i_ef = np.array([i for i, v in enumerate(pars) if 'efac' in v],dtype=np.int)
        i_eq = np.array([i for i, v in enumerate(pars) if 'equad' in v],dtype=np.int)

        m_ef = np.zeros(len(pars),dtype=np.bool)
        m_ef[i_ef] = True
        m_eq = np.zeros(len(pars),dtype=np.bool)
        m_eq[i_eq] = True

        n_w=np.logical_not(np.logical_or(m_ef,m_eq))
        imax = np.argmax(chain[:,len(pars)+1])
        pmax = chain[imax,:len(pars)]

        fig=corner.corner(post[:,n_w], labels=pars[n_w], smooth=True,truths=pmax[n_w])

        fig.savefig("%s.ms_corner.pdf"%psr.name)



        print("\n~~~~~\n")
        print("Now restrict parameters to those in best model...")
        pta = pta[bestmodel]

        pmask=np.zeros(len(pars),dtype=bool)
        for pp  in pta.param_names:
            idx = pars == pp
            pmask = np.logical_or(pmask,idx)
        pars = pars[pmask]
        pmask = np.append(pmask,[True,True,True,True])
        mask = np.rint(chain[:,-5]) == bestmodel
        chain = chain[:,:len(pmask)]
        chain = chain[mask,:]
        chain = chain[:,pmask]
        burn = int(0.25 * chain.shape[0])


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
#print("Max LogLikelihood",pta.get_lnlikelihood(pmax))
print("")
outpar=par+".post"
resfname=par+".results"
with open(resfname,"w") as resf:
    with open(outpar,"w") as outf:
        s=("{:20s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s} {:10s}".format("#param"," max-like"," mean"," std"," 2.5%"," 15.9%"," 50%"," 84.1%"," 97.5%"))

        print(s)
        resf.write(s+"\n")
        for line in parlines:
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
            elif "efac" in p:
                flag=p.split('_',1)[1]
                flag=flag[:-5]
                p="TNEF %s %s"%(selflag,flag)
            elif "ecorr" in p:
                flag=p.split('_',1)[1]
                flag=flag[:-12]
                p="TNECORR %s %s"%(selflag,flag)
                v = np.pow(10,v)*1e6 # Convert log-seconds to microseconds
            elif "equad" in p:
                flag=p.split('_',1)[1]
                flag=flag[:-12]
                p="TNEQ %s %s"%(selflag,flag)
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
            elif "_gamma" in p:
                p="TNRedGam"
            elif "log10_A" in p:
                p="TNRedAmp"
                outf.write("TNRedC %d\n"%nC)
                outf.write("TNRedFLow {}\n".format(np.log10(1.0/args.tspan_mult)))
            s = "{:20s} {:< 10g} {:< 10g} {:< 10g} {:< 10g} {:< 10g}  {:< 10g} {:< 10g} {:< 10g} ".format(p,v,mean,sig,l,l68,median,u68,u)
            resf.write(s+"\n")
            print(s)
            if p.startswith("GLTD"):
                v = np.power(10.0,v)
            if p.startswith("GLF0D"):
                v *=1e-6
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
    if psr.name+"_pm_pmra" in pars and psr.name+"_pm_pmdec" in pars:
        pos=psr.pos
        #psrdec = np.arcsin(pos[2]/np.sqrt(np.sum(pos*pos)))
        i=0
        for p in pars:
            if "pmra" in p:
                pmra = chain[burn:,i]
            if "pmdec" in p:
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
    with PdfPages("%s.chain.pdf"%psr.name) as pdf:
        for i in range(len(pars)):
            fig=plt.figure(figsize=(16,8))
            plt.plot(x[:burn],chain[:burn,i],'.',color='gray')
            plt.plot(x[burn:],chain[burn:,i],'.',color='k')
            plt.title("%s"%pars[i])
            plt.ylabel("%s"%pars[i])
            pdf.savefig()
            plt.close()



m_f2 = (pars=='F2')
i_ef = np.array([i for i, v in enumerate(pars) if 'efac' in v],dtype=np.int)
i_eq = np.array([i for i, v in enumerate(pars) if 'equad' in v],dtype=np.int)

m_ef = np.zeros(len(pars),dtype=np.bool)
m_ef[i_ef] = True
m_eq = np.zeros(len(pars),dtype=np.bool)
m_eq[i_eq] = True

n_w=np.logical_not(np.logical_or(m_ef,m_eq))
imax = np.argmax(chain[:,len(pars)])
pmax = chain[imax,:len(pars)]

plt.rcParams.update({'font.size': 8})

fig=corner.corner(post[:,n_w], labels=pars[n_w], smooth=True,truths=pmax[n_w])

fig.savefig("%s.corner.pdf"%psr.name)


nderived=len(pars_derived)-len(pars)
if nderived > 0:
    derived = np.pad(n_w,(0,nderived),'constant',constant_values=(True))
    fig=corner.corner(post_derived[:,derived], labels=pars_derived[derived], smooth=True)
    fig.savefig("%s.corner_derived.pdf"%psr.name)



if args.all_corner:
    fig=corner.corner(post, labels=pars, smooth=True);
    fig.savefig("%s.corner_all.pdf"%psr.name)



if args.white_corner:
    fig=corner.corner(post[:,m_ef], labels=pars[m_ef], smooth=True);
    fig.savefig("%s.corner_efac.pdf"%psr.name)
    fig=corner.corner(post[:,m_eq], labels=pars[m_eq], smooth=True);
    fig.savefig("%s.corner_equad.pdf"%psr.name)



if f2range > 0:
    print("F2",np.mean(post[:,m_f2])*1e-27,np.std(post[:,m_f2])*1e-27)
