import glob
import os

from astropy.io import fits
from astropy import units as u
import pylab as pl
import numpy as np

import utils

from astropy import logger
logger.log.setLevel('WARNING')
dlocals = dict()
if len(logger.log.filters) == 0:
    class myfilter(object):
        def filter(self,record):
            if 'LAT' in record.msg:
                return False
            return True
    logger.log.addFilter(myfilter())

#
#**** HAVE VERIFIED THAT LAT_WN IS ACCURATE:
#**** FOR 6-month TOAs, fs=2/yr.  Periodogram with random data with same
#**** r.m.s. and fs=2 yields (1) data rms = power (2) power = lat_wn
# e.g. J1231-1411 TOAs are about 2.3 mus error at 6-month cadence.
# periodogram(np.random.randn(240000)*2.3e-6,fs=2.0)[1][1:].mean() = 5.3e-12
# (2.3mus)**2 = 5.23e-12
# lat_wn = 6.116e-12 s^2/yr or 6 mus^2/yr.


# TODO -- join data/timing import structure
# TODO -- further checks on absence of F2/TN
# (1) select MSPs

tank_dir = '/tank/kerrm/fermi_data'
anc_dir = '/home/kerrm/research/pulsar-timing/data/ancillary'
pp_dir = tank_dir + '/pulsar_pipeline'

def check_tnfits():
    # very crude, just check to see if there is a tnfit.par for each
    # pulsar directory
    # TODO -- move this into output.py
    psrs = sorted([os.path.basename(x) for x in glob.glob(pp_dir + '/J*')])
    no_tnfits = []
    for psr in psrs:
        if not os.path.isfile(pp_dir + '/%s/timing/%s_tnfit.par'%(psr,psr)):
            no_tnfits.append(psr)
    return no_tnfits


def get_msps():

    f = fits.open(anc_dir + '/obj-pulsar-lat_v1080.fits')
    names = f[1].data.field('source_name')
    f0 = f[1].data.field('F0')
    is_msp = f0 > 1./25e-3
    jnames = np.asarray([x.split(' ')[-1] for x in names])[is_msp]
    return jnames

msps = get_msps()
tnfit_pars = ['%s/%s/timing/%s_tnfit.par'%(pp_dir,x,x) for x in msps]
tnfit_pars = [x for x in tnfit_pars if os.path.isfile(x)]

results = dict()
for par in tnfit_pars:
    lat_logl = utils.get_par_value(par,'LATLOGL')
    if (lat_logl is None):
        print('Missing entry for %s.'%par)
        continue
    lat_wn = utils.get_par_value(par,'LATWN')
    if (lat_wn is None):
        print('Missing entry for %s.'%par)
        continue
    lat_h = utils.get_par_value(par,'LATHTEST')
    if (lat_h is None):
        print('Missing entry for %s.'%par)
        continue
    jname = os.path.basename(par).split('_tnfit')[0]

    m = utils.get_pint_model(par)
    F0 = m.F0.value
    comp,order,comp_type_list,comp_type = m.map_component(
            'AstrometryEquatorial')
    ecl = comp.coords_as_ECL()
    # coordinates for hasasia -- ecliptic longitude and co-latitude in rad
    phi = ecl.lon.to(u.rad).value
    theta = (90*u.deg-ecl.lat).to(u.rad).value
    finish = m.FINISH.value

    results[jname] = [float(lat_logl),float(lat_wn),F0,float(lat_h),phi,theta,finish]

good_jnames = np.asarray(sorted(results.keys()))
lat_logls = np.asarray([results[jname][0] for jname in good_jnames])
lat_wns = np.asarray([results[jname][1] for jname in good_jnames])
f0s = np.asarray([results[jname][2] for jname in good_jnames])
htests = np.asarray([results[jname][3] for jname in good_jnames])
phis = np.asarray([results[jname][4] for jname in good_jnames])
thetas = np.asarray([results[jname][5] for jname in good_jnames])
finishes = np.asarray([results[jname][6] for jname in good_jnames])

mask = lat_logls > 0
if ~mask.sum() > 0:
    print('These pulsars had a negative log likelihood:')
    print(good_jnames[~mask])

good_jnames = good_jnames[mask]
lat_logls = lat_logls[mask]
lat_wns = lat_wns[mask]
f0s = f0s[mask]
htests = htests[mask]
phis = phis[mask]
thetas = thetas[mask]
finishes = finishes[mask]

pl.figure(1); pl.clf()
pl.rcParams['xtick.labelsize'] = 'large'
pl.rcParams['ytick.labelsize'] = 'large'
pl.clf()
pl.loglog(lat_logls,lat_wns*1e12,marker='o',ls=' ')
pl.xlabel('Log Likelihood',size='large')
pl.ylabel('White Noise ($\mathrm{\mu s}^2\,\mathrm{yr}^{-1}$)',size='large')
pl.tight_layout()
pl.savefig("lat_msps.pdf")

proxy = htests*f0s
proxy *= np.mean(lat_wns/proxy)

pl.figure(2); pl.clf()
pl.rcParams['xtick.labelsize'] = 'large'
pl.rcParams['ytick.labelsize'] = 'large'
pl.clf()
pl.loglog(proxy*1e12,lat_wns*1e12,marker='o',ls=' ')
pl.xlabel('Proxy White Noise',size='x-large')
pl.ylabel('White Noise ($\mathrm{\mu s}^2\,\mathrm{yr}^{-1}$)',size='x-large')
pl.tight_layout()
#pl.savefig("lat_msps.pdf")

# write out the data for use elsewhere
with open('lat_msps.asc','w') as f:
    f.write('# JNAME F0 LATLOGL LATWN (s^2/yr) HTEST PHI (rad) THETA (colat, rad) FINISH (MJD)\n')
    for i in range(len(good_jnames)):
        f.write('%s %.3f %.2f %.4g %.2f %.4f %.4f %.2f\n'%(good_jnames[i],f0s[i],lat_logls[i],lat_wns[i],htests[i],phis[i],thetas[i],finishes[i]))
