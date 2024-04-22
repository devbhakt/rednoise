# This test is DISABLED because event_optimize requires PRESTO to be installed
# to get the fftfit module.  It can be run manually by people who have PRESTO.
# Actually it's not disabled? Unclear what the above is supposed to mean.
import os
import shutil
import sys
from io import StringIO
from pathlib import Path

import time
import emcee.backends
import pytest
import numpy as np
import pickle
import scipy.stats as stats

import event_optimize
import event_optimize_phase

def test_phase(tmp_path):
    parfile = "/home/drb7tg/git/rednoise/phase_calc/J1231.par"
    eventfile_orig = ("J1231_srcprob.fits.pickle.gz")
    temfile = "/home/drb7tg/git/rednoise/phase_calc/1231.gaussians"
    eventfile = tmp_path / "event.fits"

    # We will write a pickle next to this file, let's make sure it's not under tests/
    shutil.copy(eventfile_orig, eventfile)


    p = Path.cwd()
    saved_stdout, sys.stdout = (sys.stdout, StringIO("_"))

    np.random.seed(1)
    event_optimize.maxpost = -9e99 
    event_optimize.numcalls = 0
    try:
        os.chdir(tmp_path)
        start_time = time.time()
        cmd = f"{eventfile} {parfile} {temfile} --minWeight=0.1 --nwalkers=20 --nsteps=100 --burnin 10 --clobber --usepickle --multicore --ncores 8 --no-autocorr"
        x = cmd.split()
        x.append('--weightcol')
        x.append('4FGL J1231.1-1412')
        event_optimize.main(x)
        end_time = time.time()

        og_eo_time = start_time - end_time
        np.random.seed(1)
        event_optimize.maxpost = -9e99
        event_optimize.numcalls = 0

        start_time = time.time()
        #cmd = f"{eventfile} {parfile} {temfile} --minWeight=0.1 --nwalkers=10 --nsteps=100 --burnin 10 --clobber --usepickle --multicore --ncores 8 --no-autocorr"
        event_optimize_phase.main(x)
        end_time = time.time()
        updated_eo = start_time - end_time

        print(f'PINT model phase calculation: \t {og_eo_time:.5f}')
        print(f'Design Matrix Phase calc: \t {updated_eo:.5f}')
        print(f'Speed up in using the Design Matrix: \t {og_eo_time/updated_eo:.5f}')

    except ImportError:
        pytest.skip(f"Pathos multiprocessing package not found")
    finally:
        os.chdir(p)
        sys.stdout = saved_stdout
