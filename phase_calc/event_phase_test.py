import time
import event_optimize
#import event_optimize_phase
import numpy as np
import pathos.multiprocessing as mp

#eventfile = 'J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_GEO_wt.gt.0.4.fits'
#parfile = 'PSRJ0030+0451_psrcat.par'
#temfile = 'templateJ0030.3gauss'

eventfile = 'J1231_srcprob.fits.pickle.gz'
parfile = 'J1231.par'
temfile = '1231.gaussians'

cmd = f"{eventfile} {parfile} {temfile} --minWeight=0.1 --nwalkers=64 --nsteps=2500 --burnin 500 --clobber --usepickle --multicore --ncores 8 --no-autocorr --basename J1231_orig_rseed --backend"
#cmd = f"{eventfile} {parfile} {temfile} --minWeight=0.1 --nwalkers=64 --nsteps=2500 --burnin 500 --clobber --usepickle --multicore --ncores 8 --no-autocorr --basename J1231_phase_calc_rseed --calc_phase --backend"

x = cmd.split()
x.append('--weightcol')
#x.append('PSRJ0030+0451')
x.append('4FGL J1231.1-1412')

print(f'Testing original event optimze')
np.random.seed(1)
#start_time = time.time()
event_optimize.main(x)
#end_time = time.time()
#og_time = end_time-start_time

#print(f'Testing updated event optimize with new phase calculation')
#start_time = time.time()
#event_optimize_phase.main(x)
#end_time = time.time()
#updated_time = end_time-start_time

#print(f'PINT model phase calculation: \t {og_time:.5f}')
#print(f'Design Matrix Phase calc: \t {updated_time:.5f}')
#print(f'Speed up in using the Design Matrix: \t {og_time/updated_time:.5f}')
