from importlib import reload
import os
import sys

from astropy import logger
import numpy as np
import pylab as pl
from scipy.integrate import simps,cumtrapz

import common_paper_code; reload(common_paper_code)
import common_paper_code as common

# Apr 24, 2021 -- super important: timing solutions *must* be converged,
# otherwise can get some wonky numerical stuff (recall 1124-3653)

# NB there is a factor of 2 difference between the Siemens et al. eq and the
# Jenet 06 version!  (This may be from the overlap reduction function and
# how it is normalized.  Need to get to the bottom of it!)

logger.log.setLevel('WARNING')
if len(logger.log.filters) == 0:
    class myfilter(object):
        def filter(self,record):
            if 'LAT' in record.msg:
                return False
            if 'NE_SW' in record.msg:
                return False
            if 'EPHVER' in record.msg:
                return False
            return True
    logger.log.addFilter(myfilter())
debug_locals = dict()

# these pulsars have numerical issues -- essentially, they can't constrain
# the additional wave terms in the marginalization.  Those with 1 harmonic
# are pretty useless, while those with 2 can do 20-30e-14.
weak_pulsars = {
        'J0312-0921': 3,
        'J0610-2100': 2,
        'J1513-2550': 2,
        'J1543-5149': 1,
        'J1741+1351': 2,
        'J1744-7619': 2,
        'J1858-2216': 1,
        'J2034+3632': 2,
    # J1741 shows some numerical issues
#        'J1843-1113': 2, # excluded by logl cut
#        'J2006+0148': 0, # excluded by logl cut
#        'J2017-1614': 1, # excluded by logl cut
#        'J2042+0246': 0, # excluded by logl cut
#        'J2052+1218': 0, # excluded by logl cut
}
#weak_pulsars = {}

strong_pulsars = {
        'J1231-1411' : 5,
        'J0614-3329' : 5,
}

outpath = '/tank/kerrm/fermi_pta/unbinned'

gwb_dom = np.logspace(-20,-10,200)
dom = np.log10(gwb_dom**2/(12*np.pi**2)*(365*86400)**2)
# marginalizing over the backgrounds
gwb_mdom = np.logspace(-20,-10,100)
#gwb_mdom = np.logspace(-20,-10,10)
mdom = np.log10(gwb_mdom**2/(12*np.pi**2)*(365*86400)**2)
idx_dom = np.arange(1,6.01,0.1)


def write_list(index_grid=False):
    """ Make list / GNU parallel commands for pulsar run."""
    logl,wn = np.loadtxt('lat_msps.asc',usecols=[2,3]).transpose()
    jnames = np.loadtxt('lat_msps.asc',usecols=[0],dtype=str).transpose()
    mask = (wn < 5e-10) & (logl > 150)
    jnames = jnames[mask]
    wn = wn[mask]
    logl = logl[mask]

    exclude_jnames = []

    mask = np.asarray([jname not in exclude_jnames for jname in jnames])
    jnames = jnames[mask]
    wn = wn[mask]
    logl = logl[mask]
    with open('limit_cmds.asc','w') as f:
        if not index_grid:
            for jname in jnames:
                f.write('python3 pta_limit.py %s\n'%jname)
        else:
            grid = np.arange(3,22)/3
            for jname in jnames:
                for idx in grid:
                    f.write('python3 pta_limit.py %s %.5f\n'%(jname,idx))

def get_lim(c,gwb_dom,val=0.95):
    cod = c
    pdf = np.exp(cod[0]-cod)
    cpdf = np.append(0,cumtrapz(pdf,x=gwb_dom))
    cpdf *= 1./cpdf[-1]
    a = np.searchsorted(cpdf,val)
    if cpdf[a] > val:
        # linear interpolation
        frac = (val-cpdf[a-1])/(cpdf[a]-cpdf[a-1])
        gwb_lim = np.exp(frac*np.log(gwb_dom[a]/gwb_dom[a-1])+np.log(gwb_dom[a-1]))
    else:
        gwb_lim = gwb_dom[a]
    return pdf,cpdf,gwb_lim

def do_one(jname,do_marg=True,clobber=False,write_output=True,
        prior_sigma=None,index=None):
    """ 
    Process GWB results for a single pulsar using a GWB model with variable
    index (specified by index, default=13/3) and an optional intrinisic
    noise process which is marginalized over using the domains specified
    above.
    """
    if index is None:
        outfile = '%s/%s_unbinned_results.asc'%(outpath,jname)
        index = 13./3
    else:
        do_marg = False
        outfile = '%s/%s_unbinned_results_%.2f.asc'%(outpath,jname,index)

    if (not clobber) and write_output and os.path.isfile(outfile):
        return

    logl,wn = np.loadtxt('lat_msps.asc',usecols=[2,3]).transpose()
    jnames = np.loadtxt('lat_msps.asc',usecols=[0],dtype=str).transpose()
    wn = wn[jnames==jname]

    d,model,(popt,popte) = common.load_data_and_model(jname,use_tnfit=True,
            quiet=True,require_tnfit=True)
    if jname in weak_pulsars:
        model.tn().add_harms(weak_pulsars[jname]-model.tn().get_nharm())
    #if jname in strong_pulsars:
        #model.tn().add_harms(strong_pulsars[jname]-model.tn().get_nharm())

    htn = model.get_htn([-15,index]) # use a notional threshold TN matrix
    LF = common.PhotonLikelihoodFit(model,d,H_tn=htn,align=False)

    cod = np.asarray([common.fit_tn_logl([a,index],LF,
        disable_check1=True,disable_check2=True,
        prior_sigma=prior_sigma) for a in dom])

    if do_marg:
        cod3 = np.empty([len(dom),len(idx_dom),len(mdom)])
        for idom,a in enumerate(dom):
            for iidx,idx in enumerate(idx_dom):
                for imdom,am in enumerate(mdom):
                    cod3[idom,iidx,imdom] = common.dual_fit_tn_logl(
                            [a,index],[am,idx],LF,
                            disable_check1=True,disable_check2=True)
        # now marginalize
        debug_locals['cod3'] = cod3
        x3 = np.exp(cod3[0,0,0]-cod3)
        cod_marg = -np.log(simps(simps(x3,axis=2,x=gwb_mdom),axis=1,x=idx_dom))
    else:
        cod_marg = np.zeros_like(cod)

    if np.any(np.isinf(cod)) or np.any(np.isinf(cod_marg)) or np.any(np.isnan(cod)) or np.any(np.isnan(cod_marg)):
        print('Failure for %s.'%jname)
        if write_output:
            with open(outfile,'w') as f:
                f.write('Failure')
                return

    lim = get_lim(cod,gwb_dom)[-1]
    if do_marg:
        limm = get_lim(cod_marg,gwb_dom)[-1]
    else:
        limm = 0

    # write out results
    # Jname WN GWB_lim1 GWB_lim2
    # gwb_dom
    # cod
    # cod_marg
    if write_output:
        with open(outfile,'w') as f:
            f.write('%s %.3f %.3f %.3f\n'%(jname,wn*1e12,lim*1e14,limm*1e14))
            f.write(' '.join(['%.5g'%x for x in gwb_dom]))
            f.write('\n')
            f.write(' '.join(['%.5f'%x for x in cod[0]-cod]))
            f.write('\n')
            f.write(' '.join(['%.5f'%x for x in cod_marg[0]-cod_marg]))
    else:
        print('Summary output for %s (not recorded): '%jname)
        print('ts = %.2f'%(cod.min()-cod[0]))
        pl.semilogx(gwb_dom,cod)
        print('lim raw %.2f'%(lim*1e14))
        print('lim marg %.2f'%(limm*1e14))

def bundle_data():
    """ Used this to round up data for zenodo.
    """
    jnames = [
            'J0030+0451',
            'J0034-0534',
            'J0101-6422',
            'J0102+4839',
            'J0312-0921',
            'J0340+4130',
            'J0418+6635',
            'J0533+6759',
            'J0613-0200',
            'J0614-3329',
            'J0740+6620',
            'J1124-3653',
            'J1231-1411',
            'J1513-2550',
            'J1514-4946',
            'J1536-4948',
            'J1543-5149',
            'J1614-2230',
            'J1625-0021',
            'J1630+3734',
            'J1741+1351',
            'J1810+1744',
            'J1816+4510',
            'J1858-2216',
            'J1902-5105',
            'J1908+2105',
            'J1939+2134',
            'J1959+2048',
            'J2017+0603',
            'J2034+3632',
            'J2043+1711',
            'J2214+3000',
            'J2241-5236',
            'J2256-1024',
            'J2302+4442']
    assert (len(jnames)==35)
    for jname in jnames:
        ft1_dir = '/tank/kerrm/fermi_pta/ft1/%s'%jname
        os.system('mkdir %s'%ft1_dir)
        os.system('cp ~/pp/%s/ft1/*fits %s'%(jname,ft1_dir))
        eph_dir = '/tank/kerrm/fermi_pta/ephemerides'
        os.system('cp ~/pp/%s/timing/%s_tnfit.par %s'%(jname,jname,eph_dir))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        import sys; sys.exit(0)
    jname = sys.argv[1]
    if len(sys.argv) > 2:
        index = float(sys.argv[2])
    else:
        index = None
    do_one(jname,clobber=True,index=index)


# convert our P(f) normalization to GWB, see e.g. Eq 2 of Jenet et al. 06
# P(f) = 1/12pi^2*1/f^3*A^2*(f/yr^1)^-4/3
# Our units are P(f) = N*(f/yr^-1)^gamma, with <N> = s^2*yr
# So to convert, we need a factor of (s/yr)^2 after setting gamma=13/3
# A_gwb^2 = 12*pi^2*N*(365*86400)**-2
# This checks out, e.g. for J1231-1411 we get a 95% limit on N of 10^-16.3
# which yields A_gwb < 2.52e-15, in good agreement with Aditya's runs.
