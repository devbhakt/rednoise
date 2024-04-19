from __future__ import division, print_function

import numpy as np

from enterprise.signals import signal_base
from enterprise.signals.parameter import function
from enterprise.signals.gp_signals import BasisGP
from enterprise.signals.selections import Selection
from enterprise.signals import selections


import enterprise.constants as const


@signal_base.function
def quasiperiodic(f, log10_A=-1, sigma=0.1, f0=0.3):
    #print("qp: ",sigma,f0)
    df = np.diff(np.concatenate((np.array([0]), f[::2])))
    return ((10**log10_A)**2 *
            np.exp(-((f/const.fyr-f0)/sigma)**2) * np.repeat(df, 2))

@function
def create_quasiperiodic_basisfunction(toas, sigma=0.1, f0=1.0, Tspan=None):
    #print("cqbf: ",sigma,f0)


    T = Tspan if Tspan is not None else toas.max() - toas.min()
    #print(T)


    fstep = 1 / T

    #print(fstep)
    #fmin = f0 - 4*sigma
    #fmax = f0 + 4*sigma

    nmodes=50

    _f0 = f0 * const.fyr

    n = min(np.round(_f0 / fstep)-1,nmodes/2)

    fmin = _f0 - n*fstep

    fmax = fmin + nmodes * fstep

    #fmin *= const.fyr
    #fmax *= const.fyr

    #if fmin < fstep:
    #    fmin=fstep

    #nmodes = np.round((fmax-fmin) / fstep)
    #if nmodes%2 == 0:
    #    nmodes += 1
    #print(nmodes, fmin/const.fyr,fmax/const.fyr)

    nmodes = int(nmodes)
    f = np.linspace(fmin,fmax, nmodes, endpoint=False)
    #print(fmin/const.fyr, fstep/const.fyr,f/const.fyr-f0)

    Ffreqs = np.repeat(f, 2)

    N = len(toas)
    F = np.zeros((N, 2 * nmodes))

    # The sine/cosine modes
    F[:,::2] = np.sin(2*np.pi*toas[:,None]*f[None,:])
    F[:,1::2] = np.cos(2*np.pi*toas[:,None]*f[None,:])

    #print("cqbf: exit")

    return F, Ffreqs


def FourierBasisGP_QP(basis, spectrum, selection=Selection(selections.no_selection), name='qp_noise'):
    """Convenience function to return a BasisGP class with a
    fourier basis for QP signal"""

    coefficients = False
    BaseClass = BasisGP(spectrum, basis, coefficients,
                        selection=selection, name=name)

    class FourierBasisGP_QP(BaseClass):
        signal_type = 'basis'
        signal_name = 'qp noise'
        signal_id = name

    return FourierBasisGP_QP
