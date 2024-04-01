from . import xparameter as parameter
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from enterprise.signals import gp_signals, signal_base, utils
import enterprise.constants as const
from enterprise.signals.selections import Selection
from enterprise.signals import selections

from enterprise.signals import deterministic_signals
from astropy.coordinates import Angle
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.coordinates import get_sun
from astropy.time import Time
from functools import partial

name="Time Variable Solar wind"
argdec="Parameters for fitting dm variations due to solar wind"

def setup_argparse(parser):
    """
    :param parser:
    :return:
    """

    parser.add_argument('-sw', '--solar-wind', action='store_true', help='Turn on solar wind')
    parser.add_argument('--sw-sigma-max', type=float, default=10, help='Max solar wind')
    parser.add_argument('--sw-sigma-min', type=float, default=1e-3, help='Min solar wind')


def fcos_full(x, phi, eclat):
    # function to fit ctheta
    return np.cos(eclat) * np.cos(2*np.pi*x/365.25 - phi)


def solar_angle(coords_psr, mjd):

    #RA Dec of the pulsar in format e.g.: 10h22m57.9992s +10d01m52.78s)                                                                                                                                            

    t = Time(mjd, format='mjd')
    sunpos = get_sun(t)
    sep = sunpos.separation(coords_psr)

    return sep   

def setup_psr(psr):

    # I *think* the psr.earth_ssb[:,3:5] etc are velocity components and I only need [:,:3]

    t2psr = psr.t2pulsar


    rsa = -t2psr.sun_ssb[:, :3] + t2psr.earth_ssb[:, :3] + t2psr.observatory_earth[:, :3]

    # r = np.sqrt(rsa * rsa)
    r = np.empty(rsa.shape[0])
    for j in range(rsa.shape[0]):
        r[j] = np.sqrt(np.dot(rsa[j], rsa[j]))

    pos = t2psr.psrPos  # this is probably corrected for velocity already since it only has size 3 (not 6)

    # ctheta = (pos * rsa) / r
    ctheta = np.empty(rsa.shape[0])
    for j in range(rsa.shape[0]):
        ctheta[j] = np.dot(pos[j], rsa[j]) / r[j]

    ''' From dm_delays.C:

    psr[p].obsn[i].freqSSB = freqf; /* Record observing frequency in barycentric frame */ (in Hz)

    '''

    freqf = t2psr.ssbfreqs()  # observing freq in barycentre frame, in Hz

    ''' From tempo2.h:

    #define AU_DIST     1.49598e11           /*!< 1 AU in m  
    #define DM_CONST    2.41e-4
    #define DM_CONST_SI 7.436e6              /*!< Dispersion constant in SI units            */
    #define SPEED_LIGHT          299792458.0 /*!< Speed of light (m/s)                       */

    '''
    AU_DIST = 1.49598e11
    DM_CONST_SI = 7.436e6
    SPEED_LIGHT = 299792458.0

    # The symmetrical, spherical solar wind, depending on observing frequency.

    spherical_solar_wind = 1.0e6 * AU_DIST * AU_DIST / SPEED_LIGHT / DM_CONST_SI * \
                           np.arccos(ctheta) / r / np.sqrt(1.0 - ctheta * ctheta) / freqf / freqf

    psr.spherical_solar_wind = spherical_solar_wind[psr.isort]

    psr.ctheta = ctheta[psr.isort]

    day_per_year = 365.25

    toas_mjds = t2psr.stoas
    
    # Fit a cos(2pi *t_mjd/day_per_year + phase) to ctheta, 
    # obtain phase and find the actual minimum of ctheta, i.e. the solar conjuction

    mjds = toas_mjds.astype(np.float64)


    # get ELAT, i.e. the max approach angle to the sun
    
    if 'RAJ' in t2psr.pars():
        
        coord_long = Angle(t2psr.vals()[t2psr.pars().index('RAJ')], unit=units.rad)
        coord_lat = Angle(t2psr.vals()[t2psr.pars().index('DECJ')], unit=units.rad)
        frame_psr = 'fk5'

    else:
        
        coord_long = Angle(t2psr.vals()[t2psr.pars().index('ELONG')], unit=units.rad)
        coord_lat = Angle(t2psr.vals()[t2psr.pars().index('ELAT')], unit=units.rad)
        frame_psr = 'barycentrictrueecliptic'

    coords_psr = SkyCoord(coord_long, coord_lat, frame=frame_psr)
    elat = coords_psr.barycentrictrueecliptic.lat



    fcos = partial(fcos_full, eclat=elat)
    popt, _ = scipy.optimize.curve_fit(fcos, mjds, ctheta, bounds=([0], [2*np.pi]))
    # phase = popt[0]

    # find the first APPROXIMATE conjuction in the data at theta = pi-ELAT
    Nfirstconj = int(mjds.min()/365.25 - 0.5 - (popt[0])/2/np.pi) + 1
    toa_solconj = 365.25 * (Nfirstconj + 0.5 + (popt[0])/2/np.pi)


    # get all the conjuctions before and after this 'minimum' one in the range of the data
    nlower = (toa_solconj - mjds.min()) / day_per_year
    nhigher = (mjds.max() - toa_solconj) / day_per_year

    conjunction_toas_mjds = np.concatenate((toa_solconj - np.arange(1, int(nlower) + 1) * day_per_year, [toa_solconj], toa_solconj + np.arange(1, int(nhigher) + 1) * day_per_year ))
    conjunction_toas_mjds.sort()
    nconj = len(conjunction_toas_mjds)

    # Now zoom in around these approximate conjunctions to find the conjunction MJD to 0.01 days:

    real_conjunction_toas_mjds = np.zeros_like(conjunction_toas_mjds)

    for i, c in enumerate(conjunction_toas_mjds):

        close_t = np.linspace(c-10, c+10, num=2000)
        close_ctheta = (-1) * np.cos(solar_angle(coords_psr, close_t))
        real_conjunction_toas_mjds[i] = close_t[np.argmin(close_ctheta)] 


    psr.sphconj_mjd = real_conjunction_toas_mjds.astype(np.float64)



def setup_model(args, psr, parfile):
    sw_model=None
    if args.solar_wind:

        setup_psr(psr)

        sw_sigma = parameter.Uniform(args.sw_sigma_min, args.sw_sigma_max, to_par=to_par)("SW_sigma")


        # Add the SW_sigma parameter to the par file
        parfile.append("NE_SW_IFUNC 0 1\n")

        for conj_toa in psr.sphconj_mjd:
            parfile.append("_NE_SW {:.2f}\n".format(conj_toa))

        sw_model = solar_wind_basis_GP(priors=sw_priors(sw_sigma=sw_sigma))


    return sw_model

@signal_base.function
def sw_priors(t, sw_sigma=1):
    """
    This is the priors on our solar wind basis function

    Probably this is constant? Or maybe there is an option for a solar cycle one?
    would be in np.ones_like(t)
    :return:
    """

    return np.ones_like(t) * sw_sigma**2


def solar_wind_basis_GP(priors, selection=Selection(selections.no_selection), name=''):

    # equivalent of utils.createfourierdesignmatrix_{red,dm}
    # RN: basis = utils.createfourierdesignmatrix_red(nmodes=components, Tspan=Tspan, modes=modes, pshift=pshift, pseed=pseed)
    # DM: basis = utils.createfourierdesignmatrix_dm(nmodes=components, Tspan=Tspan,fref=1400.0)
    basis = solar_wind_basis()
    BaseClass = gp_signals.BasisGP(priors, basis, selection=selection, name=name)

    class solar_wind_basis_class(BaseClass):
        signal_type = 'basis'
        signal_name = 'solar wind'
        signal_id = 'solar_wind_' + name if name else 'solar_wind'

    return solar_wind_basis_class

def ifunc(xi, i, x):
    y = np.zeros_like(xi)
    y[i] = 1
    return np.interp(x,xi,y)

@signal_base.function
def solar_wind_basis(toas, spherical_solar_wind, sphconj_mjd):

    """
    :param toas: vector of time series in seconds

    return the basis evaluated at toas (i.e. the solar wind model for unit NESW, for each of our parameters),
    and something else which is equivalent to the frequencies in the fourier basis

    the libstempo object is psr.t2pulsar  Calculate all this in the setup function and just pass around
     the spherical solar wind array or something.

    rsa[j] = -psr[p].obsn[i].sun_ssb[j] + psr[p].obsn[i].earth_ssb[j] + psr[p].obsn[i].observatory_earth[j];
    r = sqrt(dotproduct(rsa,rsa));
    ctheta = dotproduct(pos,rsa)/r;
    psr[p].obsn[i].spherical_solar_wind = 1.0e6*AU_DIST*AU_DIST/SPEED_LIGHT/DM_CONST_SI*acos(ctheta)/r/sqrt(1.0-ctheta*ctheta)/freqf/freqf;


    (times an "ifunc")


    :return:
    """


    """
    Get the ifunc Fourierfrequency-equivalent, which should be toas every year, close to solar conjuction.
    Solar conj happens at max spherical SW, or at cos(theta) (= ctheta) at minimum (i.e. == -1)
    """

    nconj = len(sphconj_mjd)

    sec_per_day = 24 * 3600

    """
    def ifunc(xi, i, x):
        y = np.zeros_like(xi)
        y[i] = 1
        return np.interp(x,xi,y)
    
    xi: the conjunction_toas
    i: the ith element of xi (i from 0 to nconj)
    x: the toas I think?
    """
    ifunc_comp = np.empty((nconj, len(toas)))

    for i in range(nconj):

        ifunc_comp[i] = ifunc(sphconj_mjd * sec_per_day, i, toas) * spherical_solar_wind

    """
    Figure out the ifunc bit here and multiply to give final basis:
        solar_wind_basis = spherical_solar_wind * somethingifunc
    """

    solar_wind_basis = ifunc_comp

    solar_wind_basis = solar_wind_basis.T

    return solar_wind_basis, sphconj_mjd * sec_per_day


def to_par(self, p, chain):

   if "SW_sigma" in p:
       return "CONSTRAIN NE_SW_IFUNC_SIGMA", chain
   else:
       return None

