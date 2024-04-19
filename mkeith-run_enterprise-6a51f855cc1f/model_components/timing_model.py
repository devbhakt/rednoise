import numpy as np
import sys

from enterprise.signals import deterministic_signals
from enterprise.signals import signal_base

from astropy import coordinates as coord
from astropy import units as units

from . import xparameter as parameter

import numpy.polynomial as polynomial

from collections import OrderedDict
import libstempo


name = "BasicTimingModel"
argdec = "Basic pulsar spin and astrometric parameters."


def setup_argparse(parser):
    parser.add_argument('--f2', type=float, default=0, help='range of f2 to search')
    parser.add_argument('--pm', action='store_true', help='Fit for PMRA+PMDEC')
    parser.add_argument('--px', action='store_true', help='Fit for parallax')
    parser.add_argument('--px-range', type=float, default=10, help='Max parallax to search')
    parser.add_argument('--pm-angle', action='store_true', help='Fit for PM + angle rather than by PMRA/PMDEC')
    parser.add_argument('--pm-range', type=float, default=10, help='Search range for proper motion (deg/yr)')
    parser.add_argument('--pm-ecliptic', action='store_true', help='Generate ecliptic coords for proper motion')
    parser.add_argument('--pos', action='store_true', help='Fit for position (linear fit only)')
    parser.add_argument('--pos-range', type=float, default=1, help='Search range for position (arcsec)')
    parser.add_argument('--legendre', action='store_true', help="Fit polynomial using Legendre series")
    parser.add_argument('--leg-df0', type=float, default=0, help="Max offset in f0 parameters")
    parser.add_argument('--leg-df1', type=float, default=0, help="Max offset in f1 parameters")
    parser.add_argument('--wrap',type=float,nargs='+',help="Fit for missing phase wraps at this epoch")
    parser.add_argument('--wrap-range',type=int, default = 2,help="Max number of wraps missing")
    parser.add_argument('--tm-fit-file', type=str,default=None, help="Use setup file for timing model parameters ONLY WORKS WITH Multinest/MPI")
    parser.add_argument('--fake-tm', action='store_true', help="Use a design matrix to fake the tm fit")
    parser.add_argument('--qr',action='store_true',help="use the QR decomposition for marginalisation")
    parser.add_argument('--svd',action='store_true',help="use the SVD decomposition for marginalisation")


def setup_model(args, psr, parfile):
    global orig_f2
    global orig_params
    orig_f2=0
    orig_f1=0
    orig_pmra=0
    orig_pmdec=0
    posepoch=-1
    for line in parfile:
        e = line.strip().split()
        if len(e) > 1:
            if e[0] == "F0":
                f0 = float(e[1])
            if e[0] == "F1":
                orig_f1 = float(e[1])
            if e[0] == "F2":
                orig_f2 = float(e[1])
            elif e[0] == "PEPOCH":
                pepoch = float(e[1])
            elif e[0] == "POSEPOCH":
                posepoch = float(e[1])
            elif e[0] == "PMRA":
                orig_pmra = float(e[1])
            elif e[0] == "PMDEC":
                orig_pmdec = float(e[1])
            elif e[0] == "RAJ":
                psr_ra = e[1]
            elif e[0] == "DECJ":
                psr_dec = e[1]

    psr.earthssb = psr.t2pulsar.earth_ssb
    psr.observatory_earth = psr.t2pulsar.observatory_earth

    if posepoch < 0:
        print("Assume POSEPOCH=PEPOCH")
        posepoch = pepoch

    if args.px and "PX" in psr.fitpars:
        print("ERROR: Can't fit for PX in both MCMC and least-squares")
        sys.exit(1)
    if args.pm and "PMRA" in psr.fitpars:
        print("ERROR: Can't fit for PMRA in both MCMC and least-squares")
        sys.exit(1)
    if args.pm and "PMDEC" in psr.fitpars:
        print("ERROR: Can't fit for PMDEC in both MCMC and least-squares")
        sys.exit(1)

    if args.pos and "RAJ" in psr.fitpars:
        print("ERROR: Can't fit for RAJ in both MCMC and least-squares")
        sys.exit(1)
    if args.pos and "DECJ" in psr.fitpars:
        print("ERROR: Can't fit for DECJ in both MCMC and least-squares")
        sys.exit(1)

    orig_toas = psr.t2pulsar.toas()
    issorted = np.all(orig_toas[:-1] <= orig_toas[1:])


    #if args.legendre and "F0" in psr.fitpars:
    #    print("ERROR: Can't fit for F0 and Legendre polynomial!")
    #    sys.exit(1)
    #if args.legendre and "F1" in psr.fitpars:
    #    print("ERROR: Can't fit for F1 and Legendre polynomial!")
    #    sys.exit(1)

    if args.f2 > 0 and "F2" in psr.fitpars:
        print("ERROR: Can't fit for F2 in MCMC and least-squares!")
        sys.exit(1)

    if args.pm or args.px or args.pos:
        psr_coord = coord.SkyCoord(psr_ra, psr_dec, unit=(units.hourangle, units.degree))
        psr.coord = psr_coord
        if not issorted:
            print("ERROR: Toas must be sorted or wierd things happen with paralax and proper motion")
            sys.exit(1)
        precompute_position_gradients(psr)

    components = []

    f2range = args.f2

    if args.legendre:
        domain = [psr.toas[0],psr.toas[-1]]
        poly = polynomial.polynomial.Polynomial([0, args.leg_df0/f0, args.leg_df1/2/f0, f2range/6/f0],domain=domain,window=domain)
        leg = poly.convert(kind=polynomial.legendre.Legendre, domain=domain, window=[-1, 1])
        print(leg.coef)
        if args.leg_df0 > 0:
            l1 = parameter.Uniform(-leg.coef[1],leg.coef[1],to_par=to_par)("L1")
        else:
            l1 = parameter.Constant(0)
        if args.leg_df0 > 0:
            l2 = parameter.Uniform(-leg.coef[2],leg.coef[2],to_par=to_par)("L2")
        else:
            l2 = parameter.Constant(0)
        if f2range > 0:
            l3= parameter.Uniform(-leg.coef[3],leg.coef[3],to_par=to_par)("L3")
        else:
            l3=parameter.Constant(0)
        legfit = deterministic_signals.Deterministic(fit_legendre(l1=l1,l2=l2,l3=l3))
        components.append(legfit)

    else:
        if f2range > 0:
            f2 = fit_f2(pepoch=parameter.Constant(pepoch), f0=parameter.Constant(f0),
                        valf2=parameter.Uniform(-f2range / 1e-27, f2range / 1e-27,to_par=to_par)('F2'))
            f2f = deterministic_signals.Deterministic(f2, name='f2')
            components.append(f2f)

    if args.pm:
        pmrange = args.pm_range
        pmra = parameter.Uniform(-pmrange + orig_pmra, pmrange + orig_pmra, to_par=to_par)
        pmdec = parameter.Uniform(-pmrange + orig_pmdec, pmrange + orig_pmdec, to_par=to_par)
        pm = deterministic_signals.Deterministic(
            fit_pm(posepoch=posepoch, pmra=pmra, pmdec=pmdec, orig_pmra=orig_pmra, orig_pmdec=orig_pmdec), name='pm')
        components.append(pm)

    if args.pos:
        print("XXXX")
        posrange = args.pos_range
        ## might want to scale range by cos dec or something...
        ra_off = parameter.Uniform(-posrange, posrange,to_par=to_par)
        dec_off= parameter.Uniform(-posrange, posrange,to_par=to_par)
        pos_fit = deterministic_signals.Deterministic(fit_pos(ra_off=ra_off,dec_off=dec_off),name="position")
        components.append(pos_fit)


    if args.px:
        pxp = parameter.Uniform(0, args.px_range, to_par=to_par)
        px = deterministic_signals.Deterministic(fit_px(px=pxp), name='px')
        components.append(px)

    if args.pm_angle:
        pmrange = args.pm_range
        pmamp = parameter.Uniform(0, pmrange, to_par=to_par)
        pmangle = parameter.Uniform(0, 2 * np.pi, to_par=to_par)
        pm = deterministic_signals.Deterministic(fit_pm_angle(posepoch=posepoch, amp=pmamp, angle=pmangle), name='pm')
        components.append(pm)

    if not args.wrap is None:
        for w_epoch in args.wrap:
            w = parameter.Uniform(-args.wrap_range-0.49, args.wrap_range+0.49,to_par=to_par)("WRAP_{:.2f}".format(w_epoch))
            cmp = deterministic_signals.Deterministic(fit_wrap(epoch=parameter.Constant(w_epoch),f0=parameter.Constant(f0),turns=w))
            components.append(cmp)

    if not(args.tm_fit_file is None):
        keys = np.loadtxt(args.tm_fit_file,usecols=(0),dtype=str)
        fitrange = np.loadtxt(args.tm_fit_file,usecols=(1))
        tm_params=dict()
        psr.tmparams_orig = dict()
        for p,v in zip(psr.t2pulsar.pars(which='set'), psr.t2pulsar.vals(which='set')):
            psr.tmparams_orig[p]=v
        for i,k in enumerate(keys):
            if k in psr.fitpars:
                print("ERROR: Can't fit for {} in both MCMC and least-squares".format(k))
                sys.exit(1)
            lo=-fitrange[i]
            hi=fitrange[i]
            print("Adding {} to fit {} -> {}".format(k,lo,hi))
            tm_params[k] = parameter.Uniform(lo, hi, to_par=to_par_tm)
        orig_params = psr.tmparams_orig

        if args.fake_tm:
            setup_fake_design_matrix(psr, keys=keys)
            tm = deterministic_signals.Deterministic(fake_tempo2_timing_model(keys=keys, **tm_params))
        else:
            tm = deterministic_signals.Deterministic(tempo2_timing_model(keys=keys, **tm_params))
        components.append(tm)

    if len(components) > 0:
        model = components[0]
        for m in components[1:]:
            model += m
        return model
    else:
        return None

def to_par_tm(self,p,chain):
    param = p.split("_",1)[1]
    p_param=param
    if param=="RAJ":
        p_param="RAJ_rad"
    if param=="DECJ":
        p_param="DECJ_rad"
    return p_param, chain+orig_params[param]


def to_par(self, p, chain):
    global orig_f2
    if p == "F2":
        return "F2", chain*1e-27 + orig_f2
    elif "px_px" in p:
        return "PX", chain
    elif "pm_angle" in p:
        return "PM-angle", chain
    elif "pm_amp" in p:
        return "PM", chain
    elif "pm_pmra" in p:
        return 'PMRA', chain
    elif "pm_pmdec" in p:
        return 'PMDEC', chain
    elif "position_ra" in p:
        return "dRAJ", chain
    elif "position_dec" in p:
        return "dDECJ", chain
    else:
        return p,chain


@signal_base.function
def fit_f2(toas, pepoch, f0, valf2):
    x = (toas - pepoch * 86400.0)
    return -1e-27 * valf2 * x * x * x / 6.0 / f0


@signal_base.function
def fit_pm_angle(toas, pos, earthssb, posepoch, amp, angle, orig_pmra=0, orig_pmdec=0):
    pmra = amp * np.cos(angle)
    pmdec = amp * np.sin(angle)
    return fit_pm(toas, pos, earthssb, posepoch, pmra, pmdec, orig_pmra, orig_pmdec)

@signal_base.function
def fit_wrap(toas, epoch, f0, turns):
    turns = np.round(turns)
    res = np.zeros_like(toas)
    res[toas > epoch* 86400.0] += turns/f0
    return res

@signal_base.function
def fit_px(toas, px):
    # This is just the linearised parallax function from tempo2...
    global d_px
    return px * d_px


def precompute_position_gradients(psr):
    """
    Pre-compute all the gradients needed for the proper motion and position fitting.
    This is all just the linear fit from tempo2, but we allow it to be done in the MCMC.
    A more complicated model could be implemented if needed.
    """
    global rad_per_day2mas_per_year
    global psrra
    global psrdec
    global d_ra
    global d_dec
    global d_px

    pos=psr.pos
    earthssb=psr.earthssb
    observatory_earth=psr.observatory_earth

    AULTSC = 499.00478364
    rce = earthssb[:, 0:3]
    re = np.sqrt(np.sum(rce * rce, axis=1))

    psrra = np.arctan2(pos[1], pos[0]) + 2 * np.pi
    psrdec = np.arcsin(pos[2] / np.sqrt(np.sum(pos * pos)))

    axy = rce[:, 2] / AULTSC
    s = axy / (re / AULTSC)
    deltae = np.arctan2(s, np.sqrt(1.0 - s * s))
    alphae = np.arctan2(rce[:, 1], rce[:, 0])

    # Convert radians/day to milliarcseconds per year
    # 60.0*60.0*1000.0*86400.0*365.25/24.0/3600.0
    rad_per_day2mas_per_year = (180.0 / np.pi) * 1314900000

    d_ra = re*np.cos(deltae)*np.cos(psrdec)*np.sin(psrra - alphae)
    d_dec =  re * (np.cos(deltae) * np.sin(psrdec) * np.cos(psrra - alphae) - np.sin(deltae) * np.cos(psrdec))

    #d_pmra = re * np.cos(deltae) * np.cos(psrdec) * np.sin(psrra - alphae)
    #d_pmdec = re * (np.cos(deltae) * np.sin(psrdec) * np.cos(psrra - alphae) - np.sin(deltae) * np.cos(psrdec))

    pxconv = 1.74532925199432958E-2 / 3600.0e3

    rca = earthssb[:, 0:3] + observatory_earth[:, 0:3]
    rr = np.sum(rca * rca, axis=1)

    rcos1 = np.sum(pos * rca, axis=1)
    d_px = 0.5 * pxconv * (rr - rcos1 * rcos1) / AULTSC

@signal_base.function
def fit_pm(toas, posepoch, pmra, pmdec, orig_pmra=0, orig_pmdec=0):
    # This is just the linearised proper motion function from tempo2...
    global rad_per_day2mas_per_year
    global psrdec
    global d_ra
    global d_dec

    # This will be pmra in rad/day
    v_pmra = (pmra - orig_pmra) / rad_per_day2mas_per_year / np.cos(psrdec)
    v_pmdec = (pmdec - orig_pmdec) / rad_per_day2mas_per_year

    t0 = (toas / 86400.0 - posepoch) # days

    return v_pmra * d_ra*t0 + v_pmdec * d_dec*t0


@signal_base.function
def fit_pos(toas, ra_off, dec_off):
    global d_ra # input ra
    global d_dec # input dec
    # This is just the linearised position fit from tempo2...
    # ra_off and dec_off are in arcseconds d_ra/d_dec are in radians
    return (d_ra*ra_off + d_dec*dec_off)/206264.806247

@signal_base.function
def fit_legendre(toas, l1, l2, l3):
    p = polynomial.legendre.Legendre([0,l1,l2,l3], domain=(toas[0],toas[-1]))
    return p(toas)


## This code based on enterprise extentions code, but just uses values directly.
# Doesn't work with emcee or any code using thread pools right now
@signal_base.function
def tempo2_timing_model(residuals, isort, t2pulsar, tmparams_orig, keys, **params):

    #old_res = np.double(t2pulsar.residuals().copy())
    orig_params = np.array([tmparams_orig[key] for key in keys])
    vals = []
    for p in keys:
        vals.append(params[p] + tmparams_orig[p])
    tmparams_vary = OrderedDict(zip(keys, vals))

    # set to new values
    t2pulsar.vals(tmparams_vary)
    new_res = np.double(t2pulsar.residuals().copy())

    # remmeber to set values back to originals
    t2pulsar.vals(OrderedDict(zip(keys,
                                  np.atleast_1d(np.double(orig_params)))))
    # Return the 'positive' signal that this updated timing model fits out
    # The final residual is the origina residual minus the return from this function
    # Note that this has been corrected from the enterprise extensions code
    # Note that new_res must be ordered same as residuals, and thus use the isort indexing
    return residuals - new_res[isort]
    #return old_res[isort] - new_res[isort]




def setup_fake_design_matrix(psr,keys):
    global fake_tm_design_matrix
    global fake_tm_scalefactors
    t2psr=psr.t2pulsar
    # turn fitting on for our parameters
    for key in keys:
        t2psr[key].fit = True


    full_dm = t2psr.designmatrix()

    ## Make the dm such that it indexes same as keys
    allparams = ["ZERO"]+list(t2psr.pars())
    fake_tm_design_matrix = {}
    fake_tm_scalefactors = {}
    for k in keys:
        fake_tm_design_matrix[k] = np.copy(full_dm[psr.isort,allparams.index(k)])
        fake_tm_scalefactors[k]=1 ## Seems that libstempo already scales the design matrix


    # undo setting fit on
    for key in keys:
        t2psr[key].fit = False


@signal_base.function
def fake_tempo2_timing_model(isort,residuals, t2pulsar, tmparams_orig, keys, **params):
    global fake_tm_design_matrix
    global fake_tm_scalefactors
    """
    Here we actually fake as if we were doing the tm analysis, but use a pre-determined design matrix.
    """
    # do we need to scale the parameter?
    ret=np.zeros_like(residuals)
    for k in keys:
        ret += fake_tm_scalefactors[k] * params[k] * fake_tm_design_matrix[k]


    if False:
        #### BEGIN DUMBPY MODE!!!
        if np.random.random() < 0.005:
            print("dumpy")
            ## dump 0.5% of samples
            orig_params = np.array([tmparams_orig[key] for key in keys])
            vals = []
            for p in keys:
                vals.append(params[p] + tmparams_orig[p])
            tmparams_vary = OrderedDict(zip(keys, vals))

            # set to new values
            t2pulsar.vals(tmparams_vary)
            new_res = np.double(t2pulsar.residuals().copy())

            # remmeber to set values back to originals
            t2pulsar.vals(OrderedDict(zip(keys,
                                          np.atleast_1d(np.double(orig_params)))))
            # Return the 'positive' signal that this updated timing model fits out
            # The final residual is the origina residual minus the return from this function
            # Note that this has been corrected from the enterprise extensions code
            correctret =  residuals - new_res[isort]
            outf="dump_tm_{:020x}.npz".format(int(np.random.random()*1e20))

            dm_version=[]
            t2_version=[]
            for k in keys:
                dm_version.append(fake_tm_scalefactors[k] * params[k] * fake_tm_design_matrix[k])
                newkeys=[k]
                orig_params = np.array([tmparams_orig[key] for key in keys])
                nvals = []
                for p in newkeys:
                    nvals.append(params[p] + tmparams_orig[p])
                tmparams_vary = OrderedDict(zip(newkeys, nvals))

                # set to new values
                t2pulsar.vals(tmparams_vary)
                new_res = np.double(t2pulsar.residuals().copy())

                # remmeber to set values back to originals
                t2pulsar.vals(OrderedDict(zip(keys,
                                              np.atleast_1d(np.double(orig_params)))))
                t2_version.append(residuals-new_res[isort])
                restore_res = np.double(t2pulsar.residuals().copy())[isort]

            np.savez(outf,keys=keys,ret=ret,correctret=correctret,residuals=residuals,dm_version=dm_version,t2_version=t2_version,vals=vals,restore_res=restore_res)

        ####
    return ret