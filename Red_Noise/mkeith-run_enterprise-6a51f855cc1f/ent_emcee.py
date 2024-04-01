
from enterprise.pulsar import Pulsar
from enterprise.signals import signal_base
import numpy as np
import copy

def init(_pta,_offs,_scls):
    global pta
    global offs
    global scls
    offs=_offs.copy()
    scls=_scls.copy()
    pta = copy.copy(_pta)
    #print("Init Thread")


def log_prob(p):
    global pta
    global scls
    global offs
    p *= scls
    p += offs
    lp=pta.get_lnprior(p)
    if not np.isfinite(lp):
        return -np.inf, -np.inf
    ll=pta.get_lnlikelihood(p)
    if not np.isfinite(ll):
        return -np.inf, -np.inf
    else:
        return ll+lp, lp # get rid of log prior?

