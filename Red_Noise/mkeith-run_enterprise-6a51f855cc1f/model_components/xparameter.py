import numpy as np
import enterprise.signals.parameter as parameter
import scipy.stats as stats

from enterprise.signals.parameter import Constant

def Uniform(pmin, pmax, size=None, to_par=None):
    def UniformInvTransform(self, x):
        return x * (pmax - pmin) + pmin

    p = parameter.Uniform(pmin, pmax, size)
    p.invT = UniformInvTransform
    p.to_par = to_par
    return p


def LinearExp(lo, hi, size=None, to_par=None):
    def LinearExpInvTransform(self, x):
        return np.log10(x * (10 ** hi - 10 ** lo) + 10 ** lo)

    p = parameter.LinearExp(lo, hi, size)
    p.invT = LinearExpInvTransform
    p.to_par = to_par
    return p


def Normal(mu,sigma, size=None, to_par=None):
    def NormalTransform(self, x):
        return stats.norm.ppf(x,mu,sigma)

    p = parameter.Normal(mu,sigma, size)
    p.invT = NormalTransform
    p.to_par = to_par
    return p
