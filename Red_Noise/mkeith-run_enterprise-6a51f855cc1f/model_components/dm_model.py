from . import xparameter as parameter
import numpy as np
from enterprise.signals import gp_signals, signal_base, utils
import enterprise.constants as const
from enterprise.signals.selections import Selection
from enterprise.signals import selections

from enterprise.signals import deterministic_signals

name="DM Variations"
argdec="Parameters for fitting dm variations"

def setup_argparse(parser):
    parser.add_argument('-D', '--dm', action='store_true', help='Enable DM variation search')
    parser.add_argument('--Adm-max', type=float, default=-12, help='Max log10A_DM')
    parser.add_argument('--Adm-min', type=float, default=-18, help='Min log10A_DM')
    parser.add_argument('--dm-gamma-max', type=float, default=5, help='Max gamma red')
    parser.add_argument('--dm-gamma-min', type=float, default=0, help='Min gamma red')
    parser.add_argument('--dm-ncoeff', type=int, default=None, help='Number of DM bins to use')
    parser.add_argument('--dm-prior-log', action='store_true',
                        help='Use uniform prior in log space for dm noise amplitude')
    parser.add_argument('--dm-tspan-mult', type=float, default=1, help='Multiplier for tspan for dm')
    parser.add_argument('--dm1', type=float, default=None, help="fit for DM1")
    parser.add_argument('--dm2', type=float, default=None, help="fit for DM2")



def setup_model(args, psr, parfile):
    dm=None
    if args.dm:
        if args.dm_ncoeff is None:
            nC = args.red_ncoeff
        else:
            nC = args.dm_ncoeff

        Tspan = psr.toas.max() - psr.toas.min()
        Tspan *= args.dm_tspan_mult
        nC = int(nC*args.dm_tspan_mult)

        parfile.append("TNDMC {}\n".format(nC))

        # Note that enterprise models DM as time delay at 1400 MHz for some reason...
        # So we need to scale our values for the prior, and later unscale them
        dm_scale_factor = np.log10(1400.0**-2 / 2.41e-4)
        A_min = args.Adm_min + dm_scale_factor
        A_max = args.Adm_max + dm_scale_factor

        if args.dm_prior_log:
            log10_Adm = parameter.Uniform(A_min, A_max, to_par=to_par)('DM_A')
        else:
            log10_Adm = parameter.LinearExp(A_min, A_max, to_par=to_par)('DM_A')
        gamma_dm = parameter.Uniform(args.dm_gamma_min,args.dm_gamma_max,to_par=to_par)('DM_gamma')
        pldm = powerlaw_nogw(log10_A=log10_Adm, gamma=gamma_dm)
        dm = FourierBasisGP_DM(spectrum=pldm, components=nC, Tspan=Tspan)


    if (not args.dm1 is None) or (not args.dm2 is None):
        DMEPOCH=55000
        orig_dm1=0
        orig_dm2=0
        for line in parfile:
            e = line.strip().split()
            if len(e) > 1:
                if e[0] == "DM1":
                    orig_dm1 = float(e[1])
                if e[0] == "DM2":
                    orig_dm1 = float(e[1])
                elif e[0] == "DMEPOCH":
                    DMEPOCH = float(e[1])
        if args.dm1 is None:
            dm1=parameter.Constant(0)
        else:
            dm1 = parameter.Uniform(orig_dm1 - args.dm1, orig_dm1+args.dm1,to_par=to_par)("DM1")
        if args.dm2 is None:
            dm2=parameter.Constant(0)
        else:
            dm2 = parameter.Uniform(orig_dm2-args.dm2, orig_dm2+args.dm2,to_par=to_par)("DM2")
        dm_poly = deterministic_signals.Deterministic(fit_dm_poly(dmepoch=parameter.Constant(DMEPOCH),dm1=dm1,dm2=dm2))

        if dm is None:
            dm=dm_poly
        else:
            dm+=dm_poly

    return dm

@signal_base.function
def powerlaw_nogw(f, log10_A=-16, gamma=5):
    df = np.diff(np.concatenate((np.array([0]), f[::2])))
    return ((10 ** log10_A) ** 2 *
            const.fyr ** (gamma - 3) * f ** (-gamma) * np.repeat(df, 2))

def FourierBasisGP_DM(spectrum, components=20,
                      selection=Selection(selections.no_selection),
                      Tspan=None, name=''):
    """Convenience function to return a BasisGP class with a
    fourier basis."""

    basis = utils.createfourierdesignmatrix_dm(nmodes=components, Tspan=Tspan,fref=1400.0)
    BaseClass = gp_signals.BasisGP(spectrum, basis, selection=selection, name=name)

    class FourierBasisGP_DM(BaseClass):
        signal_type = 'basis'
        signal_name = 'dm noise'
        signal_id = 'dm_noise_' + name if name else 'dm_noise'

    return FourierBasisGP_DM

def to_par(self,p,chain):
   dm_scale_factor = np.log10(1400.0 ** -2 / 2.41e-4)
   if "DM_A" in p:
      return "TNDMAmp", chain - dm_scale_factor
   if "DM_gamma" in p:
      return "TNDMGam", chain
   if "DM1" in p:
       return "DM1", chain
   if "DM2" in p:
       return "DM2", chain
   else:
       return None


@signal_base.function
def fit_dm_poly(toas, freqs, dmepoch, dm1, dm2):
    x_yr = (toas/86400.0 - dmepoch)/365.25
    delta_dm = x_yr*dm1 + 0.5*x_yr*x_yr*dm2
    t = (freqs**-2)*delta_dm/2.41e-4
    return t
