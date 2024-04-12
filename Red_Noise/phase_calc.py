# %% 
import pint.models as models
import pint.toa as toa
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from pint.scripts.event_optimize import get_fit_keyvals
# %%
m = models.get_model('J2028_wrn.par')
t = toa.load_pickle('J2028_0.5weight.pickle.gz')

# %%
D, params, units = m.designmatrix(t)
D *= -m.F0.value
D = D.transpose() # This gives me d_phase/d_param
mjds = t.get_mjds()
for i in range(1,len(units)):
    units[i] = units[i]*u.Hz
# %%
plt.plot(mjds,D[6],'+')
plt.xlabel(f'{params[6]} ({units[6]})')
# %%
fitkeys, fitvals, fiterrs = get_fit_keyvals(m, phs = 0, phserr = 0.3)
nwalkers = 1
# %%
rand_pos = [
            fitvals + fiterrs * 0.1 * np.random.randn(len(params))
            for ii in range(nwalkers)
        ]

# %%
D[1]*(fitvals[0]-rand_pos[0][0])
# %%
D[6]*(fitvals[5]-rand_pos[0][5])
# %%
phs = np.zeros(len(mjds))
for i in range(len(rand_pos[0])-1):
    print(i)
    # x = D[i+1]*(fitvals[i]-rand_pos[0][i])
    # phs = phs + x
    # print(x)
    phs += D[i+1]*(fitvals[i]-rand_pos[0][i])
# %%
plt.plot(mjds,phs)
# %%
phs = D[6]*(fitvals[5]-rand_pos[0][5])+ D[5]*(fitvals[4]-rand_pos[0][4])
# %%
