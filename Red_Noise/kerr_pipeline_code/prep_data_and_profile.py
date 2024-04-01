import argparse
import os
import pickle
import time

import numpy as np
from pint.templates import lcprimitives,lctemplate,lcfitters
from pint.plot_utils import phaseogram
from pint.eventstats import hmw
import pylab as pl

import utils
from common_paper_code import load_merged_toas

def get_initial_template(ph,we):
    best_logl = np.inf
    best_lct = None
    for i in range(5):
        g = lcprimitives.LCGaussian(p=[0.03,0.2*i])
        lct = lctemplate.LCTemplate([g],[1])
        lcf = lcfitters.LCFitter(lct,ph,weights=we,binned_bins=1000)
        lcf.fit(unbinned=False,quiet=True)
        logl = lcf.loglikelihood(lct.get_parameters())
        if logl < best_logl:
            best_lct = lct
            best_logl = logl
    return best_lct,best_logl

def add_gaussian(lct0,ph,we):
    best_logl = np.inf
    best_lct = None
    for i in range(10):
        g = lcprimitives.LCGaussian(p=[0.03,0.1*i])
        lct = lct0.add_primitive(g,0.1)
        lcf = lcfitters.LCFitter(lct,ph,weights=we,binned_bins=1000)
        lcf.fit(unbinned=False,quiet=True)
        logl = lcf.loglikelihood(lct.get_parameters())
        if logl < best_logl:
            best_lct = lct
            best_logl = logl
    for i in range(5):
        g = lcprimitives.LCGaussian(p=[0.1,0.2*i])
        lct = lct0.add_primitive(g,0.05)
        lcf = lcfitters.LCFitter(lct,ph,weights=we,binned_bins=1000)
        lcf.fit(unbinned=False,quiet=True)
        logl = lcf.loglikelihood(lct.get_parameters())
        if logl < best_logl:
            best_lct = lct
            best_logl = logl
    return best_lct,best_logl

def auto_template(ph,we,maxg=7,logl_min=12):
    print('Constructing one-gaussian initial template.')
    lct0,logl0 = get_initial_template(ph,we)
    for i in range(maxg-1):
        print('Testing %d gaussians.'%(i+2))
        lct,logl = add_gaussian(lct0,ph,we)
        if (logl0-logl) < logl_min:
            break
        lct0,logl0 = lct,logl
    return lct0

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Make diagnostic plots of output.")

    parser.add_argument("jname",help="pulsar name (e.g. J0101-6422)")
    parser.add_argument("--minweight",help="Minimum photon weight.",type=float,default=0.05)
    parser.add_argument("--maxphot",help="Maximum number of events.  Adjust minimum weight upwards until satisfied.",type=float,default=int(3e5))
    parser.add_argument("--minMJD",help="Cut data before this MJD.",type=float,default=0)
    parser.add_argument("--maxMJD",help="Cut data after this MJD.",type=float,default=60000)
    parser.add_argument("--maxg",help="Use no more than this many gaussians to model profile.",type=int,default=7)
    parser.add_argument("-p", "--parfile",help="Override ephemeris.",type=str,default='')
    parser.add_argument("--noclip",action='store_true',help="Do *not* exclude data outside of START/FINISH",default=False)

    args = parser.parse_args()

    minWeight = utils.fix_minweight(args.minweight)
    minMJD = args.minMJD
    maxMJD = args.maxMJD

    jname = args.jname.rstrip('/')
    timingdir = '/tank/kerrm/fermi_data/pulsar_pipeline/%s/timing'%jname
    outstem = timingdir + '/%s_'%jname
    ts = load_merged_toas(jname)
    weights = np.asarray(ts.table['weight']).copy()

    # TODO -- handle minimum weight!

    if len(args.parfile)==0:
        parfile = outstem + 'tnfit.par'
        if not os.path.isfile(parfile):
            parfile = outstem[:-1] + '.par'
        if not os.path.isfile(parfile):
            raise Exception('No valid parfile!')
    else:
        parfile = args.parfile
    print('Using parfile %s.'%parfile)

    # if using a TNFIT model, delete the LATPHASE entry in case it's there
    # it will be re-aligned later in the game.
    if parfile.endswith('tnfit.par'):
        lines = open(parfile,'r').readlines()
        lines = [line for line in lines if ('LATPHASE' not in line)]
        with open(parfile,'w') as f:
            for line in lines:
                f.write(line)

    if not args.noclip:
        start,finish = utils.get_bounds(parfile)
        if start is None:
            start = minMJD
        if finish is None:
            finish = maxMJD
        if minMJD is not None:
            minMJD = max(minMJD,float(start))
        else:
            minMJD = float(start)
        if maxMJD is not None:
            maxMJD = min(maxMJD,float(finish))
        else:
            maxMJD = float(finish)
        print('new min and max MJD: ',minMJD,maxMJD)
        
        mjds = ts.get_mjds().value
        mask = (mjds >= minMJD) & (mjds <= maxMJD)
        ts.select(mask)
        weights = weights[mask]

    ephem = utils.get_solar_system_ephem(parfile)
    modelin = utils.get_pint_model(parfile,remove_lowfreq_delays=False)

    t1 = time.time()
    phases = (modelin.phase(ts,abs_phase=False).frac.value).astype(np.float64)
    phases[phases <0] += 1
    t2 = time.time()
    print('Required %.2f seconds to compute phase.'%(t2-t1))
    print('H-test = %.2f'%(hmw(phases,weights)))

    # make initial template
    lct = auto_template(phases,weights,maxg=args.maxg)
    lct.minweight = minWeight
    lcf = lcfitters.LCFitter(lct,phases,weights=weights,binned_bins=1000)
    lcf.plot(nbins=100,plot_components=True)
    pl.savefig(outstem + 'auto_lcfplot.png')

    # should also probably make the template consistent?  meh...
    lcf.minweight = minWeight
    auto_fname = outstem + 'auto_template.pickle'
    pickle.dump(lct,open(auto_fname,'wb'))
    tmpl_fname = outstem + 'template.pickle'
    if not os.path.isfile(tmpl_fname):
        pickle.dump(lct,open(tmpl_fname,'wb'))

    # NB this is made in qa.py, might want to consider joining these!

    #plot_2d(lcf.template,phases,weights,tmin=minMJD,tmax=maxMJD,window=28,
    #        tstep=7,nphi=50,clip=10,output='test.png')

    plotfile = outstem + 'auto_phaseogram.png'
    phaseogram(ts.get_mjds(), phases, weights=weights, bins=100,
        rotate=0.0, size=5, alpha=0.10, plotfile=plotfile)
