# utils.py
from astropy import units as u
from collections import deque
import numpy as np

import pint.toa as toa
import pint.models
from pint.models.parameter import prefixParameter
import pint.fermi_toas as fermi

class Waves(object):

    def __init__(self,cos_comps,sin_comps,epoch,wave_om,spinfreq):
        self.cos_comps = cos_comps
        self.sin_comps = sin_comps
        self.epoch = epoch
        self.wave_om = wave_om
        self.spinfreq = spinfreq

    def to_par(self):
        # TODO
        pass

    def shift_epoch(self,new_epoch):
        # TODO I haven't checked this at all
        dt = self.epoch-new_epoch
        nfreq = len(self.cos_comps)
        dphi = self.wave_om*dt*np.arange(1,nfreq+1)
        cdphi = np.cos(dphi)
        sdphi = np.sin(dphi)
        new_cos_comps  =  self.cos_comps*cdphi + self.sin_comps*sdphi
        self.sin_comps = -self.cos_comps*sdphi + self.sin_comps*cdphi
        self.epoch = new_epoch
        self.cos_comps = new_cos_comps

    def to_model_coeffs(self,nfreq_final=None):
        nfreq = len(self.cos_comps)
        if nfreq_final is None:
            nfreq_final = nfreq
        coeffs = np.zeros(2*nfreq_final+1)
        # convert to phase
        coeffs[1:(nfreq+1)] = self.cos_comps*(-1*self.spinfreq)
        coeffs[(nfreq_final+1):] = self.sin_comps*(-1*self.spinfreq)
        return coeffs
            
def get_wavecomps(model):
    """Return the WAVE components from a TimingModel object.

    Parameters
    ----------
    model : pint.model.TimingModel
        A realization with a Wave component

    Returns
    -------
    waves : Waves
        a Waves wrapper object encapsulating the model parameters
    """
    nfreq = model.components['Wave'].num_wave_terms
    coeffs = np.empty(2*nfreq+1,dtype=float)
    cos_comps = np.empty(nfreq)
    sin_comps = np.empty(nfreq)
    for i in range(1,nfreq+1):
        q = getattr(model,'WAVE%d'%i).quantity
        cos_comps[i-1] = q[1].value
        sin_comps[i-1] = q[0].value

    epoch = getattr(model,'WAVEEPOCH').value
    wave_om = getattr(model,'WAVE_OM').value
    spinfreq = model.F0.value
    return Waves(cos_comps,sin_comps,epoch,wave_om,spinfreq)

def remove_waves(model,grid_ts=None,no_wave=False):
    """Remove WAVE terms from model and return a Waves object.
    
    Parameters
    ----------
    grid_ts : pint.toa.TOAs
        If provided, evaluate the difference in phase on the grid.
    
    """
    try:
        comp, order, comp_type_list, comp_type = model.map_component('Wave')
    except AttributeError:
        return
    wave = get_wavecomps(model)
    if grid_ts is not None:
        old_phs = model.phase(grid_ts)
    # remove the WAVE component
    comp_type_list.pop(order)
    if grid_ts is not None:
        new_phs = model.phase(grid_ts)
        dphs = old_phs-new_phs
        dphs = dphs.int.value + dphs.frac.value
        if no_wave:
            return dphs
        return wave,dphs
    return wave

def remove_ifuncs(model):
    try:
        comp, order, comp_type_list, comp_type = model.map_component(
                'IFunc')
    except AttributeError:
        return
    comp_type_list.pop(order)

def remove_rococo_glitches(model):
    """ Remove the GLF2 parameters from all glitches."""
    try:
        comp, order, comp_type_list, comp_type = model.map_component(
                'Glitch')
        nglitch = len(comp.param_prefixs['GLEP_'])
    except AttributeError:
        return
    for i in range(1,nglitch+1):
        p = getattr(comp,'GLF2_%d'%i)
        p.value = 0
        p.frozen = True

def remove_taylor(model,grid_ts=None,H=None,max_f=1):
    """Remove higher order FN terms from a model.
    
    than F1 from model and evaluate the
     difference in phase on the gridded TOAs.  If H is provided,
        return the project the phase difference onto the Fourier
        coefficients and return them.

    Parameters
    ----------
    grid_ts : pint.toa.TOAs
        If provided, evaluate the difference in phase on the grid.
        REQUIRED FOR USE WITH H.
    H : ndarray [nsin/cos x ngrid]
        If provided, use H to project dphi onto Fourier amplitudes.
    max_f : int
        The maximum FN term to leave in the model (max_f=1 <--> F0 + F1)
    """
    comps = set(model.components['Spindown'].params)
    low_freq = ['PEPOCH'] + ['F%d'%i for i in range(max_f+1)]
    comps_highfreq = comps.difference(low_freq)
    if len(comps_highfreq) == 0:
        return None
    if grid_ts is not None:
        old_phs = model.phase(grid_ts)
    for comp in comps_highfreq:
        model.components['Spindown'].params.remove(comp)
    model.components['Spindown'].num_spin_terms = max_f + 1
    if grid_ts is not None:
        new_phs = model.phase(grid_ts)
        dphs = old_phs-new_phs
        dphs = dphs.int.value + dphs.frac.value
        if H is not None:
            coeffs = (H*dphs[None,:]).mean(axis=1)*2
            coeffs[0] *= 0.5 # correct for single real channel
            return coeffs
        return dphs

def add_spindown_component(model,degree=None,free=True,quiet=True):
    """ E.g. add F2 to a model.
    
    Default behavior is to add the "next highest" degree, e.g. add an
    F2 if F0 and F1 are already present.  If the degree is specified, will
    attempt to add it, but it must also be the "next highest".  In this
    case, however, attempting to add F2 explicitly to a model with F2
    already will result in no change.
    """
    comp, order, comp_type_list, comp_type = model.map_component(
            'Spindown')
    if degree is None:
        degree = comp.num_spin_terms
    if degree < comp.num_spin_terms:
        # already present in model
        if not quiet:
            print('Specified term F%d already in model.'%degree)
        # free it if it's not free and we want to
        if free:
            getattr(comp,'F%d'%degree).frozen = False
        return
    if degree > comp.num_spin_terms:
        if not quiet:
            print('Specified term F%d cannot be added to model.'%degree)
        return
    p = prefixParameter(name='F%d'%degree,value=0.0,uncertainty=0.0,
            units=u.Hz/(u.s)**degree,frozen=not free)
    comp.add_param(p)
    comp.setup()

def add_glitch_component(p,model):
    """
    Add glitch parameters to PINT model.

    model = a PINT timing model
    p = GLEP, GLPH, GLF0, GLF1
    """

    try:
        comp, order, comp_type_list, comp_type = model.map_component(
                'Glitch')
        has_glitch = True
        glidx = 1 + len(comp.param_prefixs['GLEP_'])
    except AttributeError:
        order = 0
        has_glitch = False
        glidx = 1

    glep = prefixParameter(name='GLEP_%d'%glidx,units='day',
            type_match='MJD',time_scale='tdb')
    glep.quantity = p[0]
    glph = prefixParameter(name='GLPH_%d'%glidx,units='pulse phase',
            value=p[1],type_match='float')
    glf0 = prefixParameter(name='GLF0_%d'%glidx,units='Hz',
            value=p[2],type_match='float')
    glf1 = prefixParameter(name='GLF1_%d'%glidx,units='Hz/s',
            value=p[3],type_match='float')

    if not has_glitch:
        comp = glitch.Glitch()
        model.add_component(comp)

    comp.add_param(glep)
    comp.add_param(glph)
    comp.add_param(glf0)
    comp.add_param(glf1)
    # add filler parameters
    """
    comp.add_param(
        prefixParameter(name='GLF2_%d'%glidx,units='Hz/(s^2)',
            value=0,type_match='float')
        )
    comp.add_param(
        prefixParameter(name='GLF0D_%d'%glidx,units='Hz',
            value=0,type_match='float')
        )
    comp.add_param(
        prefixParameter(name='GLTD_%d'%glidx,units='d',
            value=0,type_match='float')
        )
    """
    comp.setup()

def get_glitch_epochs(model):
    """ model -- a PINT timing model object"""
    epochs = []
    try:
        comp,order,comp_type_list,comp_type = model.map_component('Glitch')
        gleps = comp.param_prefixs['GLEP_']
        for glep in gleps:
            epochs.append(getattr(comp,glep).value)
    except AttributeError:
        pass
    return epochs

def get_toas(eventfile,weightcol=None,target=None,usepickle=True,
        minweight=0,minMJD=0,maxMJD=100000,ephem='DE421',clobber=False):

    # TODO: make this properly handle long double
    if usepickle and not clobber:
        try:
            picklefile = toa._check_pickle(eventfile)
            ts = toa.TOAs(picklefile)
            print('unpickled %s'%(picklefile))
            return ts
        except:
            pass
    # Read event file and return list of TOA objects
    tl = fermi.load_Fermi_TOAs(eventfile, weightcolumn=weightcol,
                               targetcoord=target, minweight=minweight,
                               minmjd=minMJD,maxmjd=maxMJD)
    # Limit the TOAs to ones in selected MJD range
    tl = filter(
            lambda t: (t.mjd.value > minMJD) and (t.mjd.value < maxMJD), tl)
    tl = list(tl)

    print("There are %d events we will use"%(len(tl)))
    # Now convert to TOAs object and compute TDBs and posvels
    ts = toa.TOAs(toalist=tl)
    if not any(["clkcorr" in f for f in ts.table["flags"]]):
        ts.apply_clock_corrections(
            include_gps=False,
            include_bipm=False,
            bipm_version=None,
        )
    ts.filename = eventfile
    ts.compute_TDBs(ephem=ephem)
    ts.compute_posvels(ephem=ephem, planets=False)
    if usepickle:
            ts.pickle()
    return ts

def get_pint_model(parfile,remove_lowfreq_delays=True):
    """A wrapper for pint.models.get_model.

    The sole difference is the default removal of all low-frequency
    model components.

    Parameters
    ---------
    remove_lowfreq_delays : bool
        Remove some low-frequency (radio) delays that are irrelevant for high-
        energy applications and expensive to calculate.

    """
    # Read in initial model
    modelin = pint.models.get_model(parfile)
    if not remove_lowfreq_delays:
        return modelin

    # get rid of low-frequency stuff
    for key in ['DispersionDM','SolarWindDispersion','TroposphereDelay']:
        for icomp,comp in enumerate(modelin.DelayComponent_list):
            if comp.__class__.__name__ == key:
                modelin.DelayComponent_list.pop(icomp)
    return modelin

def get_par_value(parfile,key,parfile_is_lines=False):
    """ Return a key/val pair from an ephemeris.

    Parameters
    ----------
    parfile : str
        File name of .par file.
    key : str
        Parameter to return, e.g. EPHEM, F0, LATPHASE.
    """
    if not parfile_is_lines:
        lines = open(parfile).readlines()
    else:
        lines = parfile
    for line in lines:
        toks = line.strip().split()
        if len(toks) < 1:
            continue
        if toks[0][0] == '#':
            continue
        if (key == toks[0]):
            return toks[1]
    return None

def get_solar_system_ephem(parfile):
    return get_par_value(parfile,'EPHEM')

def get_bounds(parfile):
    """Return START and FINISH from parfile.
    
    Returns
    -------
    start : None or str
        START value from ephemeris
    finish : None or str
        FINISH value from ephemeris
    """
    start = None
    finish = None
    lines = open(parfile).readlines()
    for line in lines:
        toks = line.split()
        if len(toks) > 0:
            if toks[0] == 'START':
                start = toks[1]
                continue
            if toks[0] == 'FINISH':
                finish = toks[1]
                continue
    return start,finish

def tempo2_as_parfile(modelin):
    """Add any missing tempo2 information to modelin."""
    pf = modelin.as_parfile()
    return pf + 'UNITS TDB\n'

def lines_replace_value(lines,key,newval):
    add_line = True
    new_lines = deque()
    for line in lines:
        if key in line:
            add_line = False
            new_lines.append('%-15s %25s'%(key,newval))
        else:
            new_lines.append(line)
    if add_line:
        new_lines.append('%-15s %25s'%(key,newval))
    return list(new_lines)

def met2mjd(times,mjdref=51910+7.428703703703703e-4):
    times = np.asarray(times,dtype=np.float128)
    return times*(1./86400)+mjdref

def mjd2met(times,mjdref=51910+7.428703703703703e-4):
    times = np.asarray(times,dtype=np.float128)
    return (times-mjdref)*86400

def fix_minweight(minweight):
    if minweight is None:
        return None
    #if minweight < 0.8:
        #return round(minweight*100)*0.01
    return round(minweight*1000000)*0.000001

def mad(v):
    return 1.4826*np.median(np.abs(np.median(v)-v))

def mad_median(v):
    m = np.median(v)
    return m,1.4826*np.median(np.abs(m-v))
