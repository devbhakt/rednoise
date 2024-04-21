# %% 
import pint.models as models
import pint.toa as toa
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import time
from pint.scripts.event_optimize import get_fit_keyvals
# %%
def main():
    print(f'Loading in model and events')
    m = models.get_model('J2028_wrn.par')
    t = toa.load_pickle('J2028_all.pickle.gz')

    print(f'Calculating the Design Matrix')
    D, params, units = m.designmatrix(t)
    D *= -m.F0.value
    D = D.transpose() # This gives me d_phase/d_param
    mjds = t.get_mjds()
    for i in range(1,len(units)):
        units[i] = units[i]*u.Hz

    fitkeys, fitvals, fiterrs = get_fit_keyvals(m, phs = 0, phserr = 0.3)
    nwalkers = 1

    rand_pos = [
                fitvals + fiterrs * 0.1 * np.random.randn(len(params))
                for ii in range(nwalkers)
            ]

    print(f'Calculating the phase using PINT Model')
    start_time = time.time()
    orig_phs = m.phase(t).frac % 1
    end_time = time.time()

    model_dt = (end_time-start_time)*u.s

    print(f'Calculating the phase using the Design Matrix')
    start_time = time.time()
    phs = np.zeros(len(mjds))
    for i in range(len(rand_pos[0])-1):
        phs += D[i+1]*(fitvals[i]-rand_pos[0][i])
    new_phase = (orig_phs+phs) % 1
    end_time = time.time()
    design_dt = (end_time-start_time) * u.s

    print(f'PINT model phase calculation: \t {model_dt:.5f}')
    print(f'Design Matrix Phase calc: \t {design_dt:.5f}')
    print(f'Speed up in using the Design Matrix: \t {model_dt/design_dt:.5f}')

if __name__ == '__main__':
    main()