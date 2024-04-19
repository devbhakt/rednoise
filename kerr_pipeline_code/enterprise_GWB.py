import numpy as np
import os, glob, json,sys
import matplotlib.pyplot as plt
import corner
import argparse

import enterprise
from enterprise.pulsar import Pulsar
from enterprise.signals import utils
from enterprise_extensions import models, model_utils


from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise.signals.parameter import function

import pandas as pd

#==========================================================================================================

parser=argparse.ArgumentParser(description="Enterprise GWB analysis")
parser.add_argument("-results",dest="results_path",help="Path to the results directory")
parser.add_argument("-bestmodels", dest="bestmodels", help="Path to the bestmodels list file")
parser.add_argument("-freespectral", dest="freespectral", help="Enable free GWB spectral model")
parser.add_argument("-noHD", dest="noHD", help="No Hellings-Downs overlap reduction function")
parser.add_argument("-noSSE", dest="noSSE", help="No BayesEphem modelling of SSE")
parser.add_argument("-noTM", dest="noTM", help="No timing model marginalisation")
parser.add_argument("-output", dest="outputdir", help="Path to the output directory")
parser.add_argument("-detection", dest="detection", help="Detection run")
parser.add_argument("-fake", dest="fake", help="Using fake toas for testing")
parser.add_argument("-cadence", dest="cadence", help="Select which cadence dataset to analyse")
args=parser.parse_args()


#===========================================Loading Par & Tim===============================================================

#Get bestmodel from temponest

if args.cadence == "yes":
    bestmodel_df = pd.read_csv(str(args.bestmodels), names=["psrname","bestmodel","cadence"], delim_whitespace=True, dtype=str)
elif args.cadence == "no":
    bestmodel_df = pd.read_csv(str(args.bestmodels), names=["psrname","bestmodel"], delim_whitespace=True, dtype=str)

path = str(args.results_path)

parfiles = []
timfiles = []
psrnames = []
bestmodels = []

if args.fake == "yes":
    print "Using fake toas for GWB search"

for index, row in bestmodel_df.iterrows():
    psrname = row["psrname"]
    bestmodel = row["bestmodel"]
    if args.cadence == "yes":
        cadence = row["cadence"]

    if args.cadence == "yes":
        bestmodel_path = os.path.join(path,"{0}/{1}C/{2}".format(psrname,cadence,bestmodel))
    elif args.cadence == "no":
        bestmodel_path = os.path.join(path,"{0}/{1}".format(psrname,bestmodel))

    if args.fake == "no":
        psrpath = os.path.join(path,"{0}".format(psrname))
        parfile = glob.glob(os.path.join(bestmodel_path,"Red*par"))[0]
        #parfile = glob.glob(os.path.join(psrpath,"{0}_TNest.par".format(psrname)))[0]
    elif args.fake == "yes":
        parfile = glob.glob(os.path.join(bestmodel_path,"*.par"))[0]

    
    if args.cadence == "yes":
        timfile = glob.glob(os.path.join(path,"{0}/*{1}*pn.tim".format(psrname,cadence)))[0]
    elif args.cadence == "no":
        timfile = glob.glob(os.path.join(path,"{0}/*.tim".format(psrname)))[0]

    print "Loading parfile: {0}".format(parfile)
    print "Loading timfile: {0}".format(timfile)

    parfiles.append(parfile)
    timfiles.append(timfile)
    psrnames.append(psrname)
    bestmodels.append(bestmodel)


#===========================================Initialising pulsars===============================================================

pulsars = []

for par, tim in zip(parfiles,timfiles):
    psr = Pulsar(par,tim,ephem="DE421") #DE421 is what's used in the LAT pipeline
    pulsars.append(psr)

tmin = [p.toas.min() for p in pulsars]
tmax = [p.toas.max() for p in pulsars]

Tspan = np.max(tmax) - np.min(tmin)

#==========================================Prior ranges================================================================

# define selection 
selection = selections.Selection(selections.no_selection)

# white noise prior ranges
efac_uniform = parameter.Uniform(0.1,5.0)
equad_uniform = parameter.Uniform(-9,-5)

efac_constant = parameter.Constant(1)
equad_constant = parameter.Constant(-20)

# red noise parameters
RN_Amp = parameter.LinearExp(-18,-9)
RN_Gamma = parameter.Uniform(0, 7)

#GWB parameters
GWB_Gamma_Uniform = parameter.Uniform(0,7)('gw_gamma_uniform')
GWB_Gamma_Constant = parameter.Constant(4.33)('gw_gamma')
if args.detection == "no":
    print "Using LinearExp prior for GWB Amp"
    GWB_Amp = parameter.LinearExp(-18,-9)('gw_log10_A')
elif args.detection == "yes":
    print "Using Uniform prior for GWB Amp"
    GWB_Amp = parameter.Uniform(-20,-11)('gw_log10_A')

#=======================================Signals===================================================================


#Signals
# Constant white noise
ef_constant = white_signals.MeasurementNoise(efac=efac_constant, selection=selection)
eq_constant = white_signals.EquadNoise(log10_equad=equad_constant, selection=selection)

# Uniform white noise
ef_uniform = white_signals.MeasurementNoise(efac=efac_uniform, selection=selection)
eq_uniform = white_signals.EquadNoise(log10_equad=equad_uniform, selection=selection)

# red noise (powerlaw with 30 frequencies)
RNPL = utils.powerlaw(log10_A=RN_Amp, gamma=RN_Gamma)
rednoise = gp_signals.FourierBasisGP(spectrum=RNPL, components=12, Tspan=Tspan)

if args.freespectral == "yes":
    print "Using a free GWB spectral index"
    gw_gamma = GWB_Gamma_Uniform
elif args.freespectral == "no":
    print "Using a constant GWB spectral index (4.33)"
    gw_gamma = GWB_Gamma_Constant

gw_pl = utils.powerlaw(log10_A=GWB_Amp, gamma=gw_gamma)

gwb_noHD = gp_signals.FourierBasisGP(spectrum=gw_pl, components=12, Tspan=Tspan, name='gw_noHD')

orf_hd = utils.hd_orf()
gwb_HD = gp_signals.FourierBasisCommonGP(spectrum=gw_pl, orf=orf_hd, components=12, Tspan=Tspan, name='gw')

eph = deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

timingmodel = gp_signals.TimingModel(use_svd=True)


#================================Model selection==========================================================================


all_signals = []

for psr,name,model in zip(pulsars,psrnames,bestmodels):

    print "======================================================================================="

    psrname = name
    bestmodel = model

    if "noWN" in bestmodel:
        print "{0}:{1} -> Not fitting for white noise. Setting EFAC to 1 and EQUAD to 0".format(psrname,bestmodel)
        signal = ef_constant + eq_constant
    
    else:
        print "{0}:{1} -> Including uniform white noise".format(psrname, bestmodel)
        signal = ef_uniform + eq_uniform

    
    if "noRN" in bestmodel:
        print "{0}:{1} -> Not fitting for a power-law red noise process".format(psrname,bestmodel)
    else:
        print "{0}:{1} -> Fitting for a power-law red noise process".format(psrname,bestmodel)
        signal = signal+rednoise


    if args.noHD == "yes":
        print "No HD ORF used"
        # no spatial correlation
        signal = signal+gwb_noHD

    elif args.noHD == "no":
        print "Using a HD ORF in the GWB search"
        # with spatial correlations
        signal = signal+gwb_HD

    if args.noSSE == "yes":
        print "No BayesEphem used to model uncertainties in SSE"
    elif args.noSSE == "no":
        print "BayesEphem enabled".format(psrname,bestmodel)
        # to add solar system ephemeris modeling...
        signal = signal+eph

    if args.noTM == "yes":
        print "No timing model marginalisation included"
    elif args.noTM == "no":
        print "Timing model marginalisation enabled"
        # timing model marginalisation
        signal = signal+timingmodel


    all_signals.append(signal(psr))
    
    print "======================================================================================="



#======================================PTA====================================================================


print "Initialising the PTA object"
# intialize PTA
   
pta = signal_base.PTA(all_signals)

print "Parameters used in the model: {0}".format(pta.param_names)

print "=========================================================================================================="


#=====================================Sampler settings=====================================================================

print "Starting sampling"
# Set initial parameters drawn from prior
x0 = np.hstack(p.sample() for p in pta.params)
ndim = len(x0)

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)


if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)

sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, outDir=str(args.outputdir), resume=True)

np.savetxt(os.path.join(str(args.outputdir),"paramnames.txt"), pta.param_names, fmt='%s')


#=======================================Sampling===================================================================
# ### Sample!

# sampler for N steps
N = int(1e6)

jp = model_utils.JumpProposal(pta)
sampler.addProposalToCycle(jp.draw_from_prior,15)
sampler.addProposalToCycle(jp.draw_from_gwb_log_uniform_distribution,10)
sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)
