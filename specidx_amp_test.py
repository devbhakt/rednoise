# %%
import numpy as np
import scipy.optimize as op
import pint.logging
import pint.simulation as ps
import pint.models as models
import pint.residuals as pr
import astropy

from pint.models import get_model
from pint.simulation import make_fake_toas_uniform
from pint.fitter import Fitter
from pint.logging import setup as setup_log
from pint.models import parameter as pp
from io import StringIO
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.visualization import quantity_support
from scipy.linalg import cho_solve,cho_factor,solve_triangular,LinAlgError
from loguru import logger as log
from scipy import signal

pint.logging.setup(level='INFO')
quantity_support
# %%
m = get_model(
   StringIO(
        """
        RAJ    		05:00:00
        DECJ   		20:00:00
        F0     		300     1 
        F1     		-1e-14  1
        PEPOCH 		58500
        #DM     		15
        #EFAC tel gbt 	1.3
        TNRedAmp 	-9.5
        TNRedGam 	4.0
        TNRedC 		30
        """	
    )
)
t = make_fake_toas_uniform(
    57000, 
    60000, 
    100, 
    model=m, 
    error=1 * u.ms, 
    #freq=u.Quantity([1400*u.MHz, 2500*u.MHz]), 
    add_noise=True, 
    add_correlated_noise=m.has_correlated_errors
)
# %%
ftr = Fitter.auto(t, m)
ftr.fit_toas()
# %%
plt.errorbar(t.get_mjds(), ftr.resids.phase_resids, ls="", marker="+")
plt.ylabel("phase")
plt.xlabel("mjd")
plt.show()
# %%
plt.errorbar(t.get_mjds(), ftr.resids.time_resids, ftr.resids.get_data_error(), ls="", marker="+")
plt.ylabel("resids (s)")
plt.xlabel("mjd")
plt.show()
# %%
# Testing the periodogram fit
f, Pxx_den = signal.periodogram(ftr.resids.time_resids,fs=1/len(ftr.resids.time_resids))
plt.semilogy(f[1:], Pxx_den[1:],'x')
plt.ylim([1e-7,0])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.xscale('log')
plt.yscale('log')
plt.show()
# %%

# %%
f, Pxx_den = signal.welch(ftr.resids.time_resids,fs=1/len(ftr.resids.time_resids))
plt.semilogy(f[1:], Pxx_den[1:],'x')
plt.ylim([1e-7,0])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.xscale('log')
plt.yscale('log')
plt.show()
# %%
