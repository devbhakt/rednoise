from . import xparameter as parameter
import numpy as np
from enterprise.signals import gp_signals, gp_bases, utils
from enterprise.signals.selections import Selection
from enterprise.signals import selections

name="Chromatic (non-dm) Variations"
argdec="Parameters for fitting non-dm chromatic variations"

def setup_argparse(parser):
    parser.add_argument('--chrom', action='store_true', help='Enable Chrom variation search')
    parser.add_argument('--Achrom-max', type=float, default=-12, help='Max log10A_Chrom')
    parser.add_argument('--Achrom-min', type=float, default=-18, help='Min log10A_Chrom')
    parser.add_argument('--chrom-index', type=float, default=4, help='Chromatic index (default=4)')
    parser.add_argument('--chrom-gamma-max', type=float, default=5, help='Max gamma red')
    parser.add_argument('--chrom-gamma-min', type=float, default=0, help='Min gamma red')
    parser.add_argument('--chrom-ncoeff', type=int, default=None, help='Number of Chrom bins to use')
    parser.add_argument('--chrom-prior-log', action='store_true',
                        help='Use uniform prior in log space for chromatic noise amplitude')
    parser.add_argument('--chrom-tspan-mult', type=float, default=1, help='Multiplier for tspan for chromatic noise')



def setup_model(args, psr, parfile):
    chrom=None
    if args.chrom:
        if args.chrom_ncoeff is None:
            nC = args.red_ncoeff
        else:
            nC = args.chrom_ncoeff

        Tspan = psr.toas.max() - psr.toas.min()
        Tspan *= args.chrom_tspan_mult
        nC = int(nC*args.chrom_tspan_mult)

        parfile.append("TNChromC {}\n".format(nC))
        parfile.append("TNChromIdx {}\n".format(args.chrom_index))

        A_min = args.Achrom_min
        A_max = args.Achrom_max

        if args.chrom_prior_log:
            log10_Achrom = parameter.Uniform(A_min, A_max, to_par=to_par)('Chrom_A')
        else:
            log10_Achrom = parameter.LinearExp(A_min, A_max, to_par=to_par)('Chrom_A')

        gamma_chrom = parameter.Uniform(args.chrom_gamma_min,args.chrom_gamma_max, to_par=to_par)('Chrom_gamma')
        plchrom = utils.powerlaw(log10_A=log10_Achrom, gamma=gamma_chrom)
        chrom = FourierBasisGP_Chrom(spectrum=plchrom, components=nC, Tspan=Tspan, index=args.chrom_index)

    return chrom

def FourierBasisGP_Chrom(spectrum, components=20,
                      selection=Selection(selections.no_selection),
                      Tspan=None, name='', index=4):
    """Convenience function to return a BasisGP class with a
    fourier basis."""

    basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=components, Tspan=Tspan,idx=index)
    BaseClass = gp_signals.BasisGP(spectrum, basis, selection=selection, name=name)

    class FourierBasisGP_DM(BaseClass):
        signal_type = 'basis'
        signal_name = 'chromatic noise'
        signal_id = 'chrom_noise_' + name if name else 'chrom_noise'

    return FourierBasisGP_DM

def to_par(self,p,chain):
   if "Chrom_A" in p:
      return "TNChromAmp", chain
   if "Chrom_gamma" in p:
      return "TNChromGam", chain
   else:
       return None

