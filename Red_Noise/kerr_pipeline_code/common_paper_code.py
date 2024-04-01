from collections import deque
from copy import deepcopy
import glob
import os
import pickle
from tempfile import mkstemp
import time

import astropy.units as u
from astropy.io import fits
import numpy as np
from scipy.integrate import simps
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_solve,cho_factor,solve_triangular,LinAlgError
from scipy.linalg import solve as mat_solve
from scipy.linalg import lstsq as mat_lstsq
from scipy.optimize import minimize,fmin

from pint import toa
from pint.templates import lcfitters
from pint.models import wave,glitch
from pint.models.parameter import prefixParameter,MJDParameter
from pint.observatory import get_observatory 
from pint.observatory.satellite_obs import get_satellite_observatory

import glitch
import utils
import tn_utils

basedir = '/tank/kerrm/fermi_data/pulsar_pipeline'
debug_locals = dict()


def pint_toas_from_mjds(sample_mjds,ephem,site='bary',errors=None,
        pulse_number=None,fake_clk_corr=True):
    freq = np.zeros(len(sample_mjds)) * u.MHz
    if errors is None:
        errors = np.zeros(len(freq)) * u.s
    if site == 'geo':
        scale = 'tt' 
    elif site=='bary':
        scale = 'tdb'
    elif site == 'Fermi':
        raise NotImplementedError('No unique FT2 at this part of the code.')
        #scale = 'tt'
        #try:
            #get_observatory('Fermi')
        #except KeyError:
            #get_satellite_observatory('Fermi',
                    #'/data/kerrm/photon_data/ft2.fits')
    else:
        raise NotImplementedError('Site = %s not supported.'%site)
    if pulse_number is None:
        tl = [toa.TOA(t,obs=site,freq=f,error=err,scale=scale) for t,f,err in zip(sample_mjds,freq,errors)]
    else:
        tl = [toa.TOA(t,obs=site,freq=f,error=err,scale=scale,flags=dict(pn=p)) for t,f,err,p in zip(sample_mjds,freq,errors,pulse_number)]
    """
    # TMP
    grid_ts = toa.get_TOAs_list(tl,include_gps=False,include_bipm=False,
            planets=False,ephem=ephem)
    grid_mjds = grid_ts.get_mjds().value
    return grid_ts,grid_mjds
    """

    grid_ts = toa.TOAs(toalist=tl)
    if fake_clk_corr:
        grid_ts.clock_corr_info['include_bipm'] = False
        grid_ts.clock_corr_info['include_gps'] = False
    else:
        grid_ts.apply_clock_corrections(
                include_gps=False,include_bipm=False)
    grid_ts.compute_TDBs(ephem=ephem)
    grid_ts.compute_posvels(ephem=ephem,planets=False)
    grid_mjds = grid_ts.get_mjds().value
    return grid_ts,grid_mjds

def geo_tzrmjd(tzrmjd,ephem,include_gps=True,include_bipm=False):
    """ Return a suitable TOAs object to be a geocentric TZRMJD.

    NB that this is a UTC time per PINT etc. convention!

    Parameters
    ----------

    tzrmjd : float
        MJD value to be used.
    ephem : str
        Solar system ephemeris to use.
    """
    tl = [toa.TOA(tz,obs='geo',freq=0*u.MHz,error=0.0,scale='utc') for tz in np.atleast_1d(tzrmjd)]
    ts = toa.TOAs(toalist=tl)
    ts.apply_clock_corrections(include_gps=include_gps,include_bipm=include_bipm)
    ts.compute_TDBs(ephem=ephem)
    ts.compute_posvels(ephem=ephem,planets=False)
    return ts

def get_tdbs(ts,model):
    """ Convert times to TDB evaluated at the barycenter.
    """
    delay = model.delay(ts).to(u.day).value
    return np.asarray(ts.table['tdbld']) - delay

class TNMatrix(object):
    """ Encapsulate the spectral decomposition form of a TN process.
    """
    def __init__(self,p,freqs,zeropad=None):
        """ NB freqs should be in cycles/year!."""
        self._freqs = freqs
        n = zeropad + 2*len(freqs)
        self._H = np.zeros((n,n),dtype=float)
        x,y = np.diag_indices_from(self._H)
        self._x = x[zeropad:]
        self._y = y[zeropad:]
        self.update_p(p)

    def update_p(self,p):
        tn_vals = 2./tn_utils.eval_pl(p,self._freqs)
        self._H[self._x[::2],self._y[::2]] = tn_vals
        self._H[self._x[1::2],self._y[1::2]] = tn_vals
        self._p = p

    def C(self):
        """ Return covariance matrix."""
        raise NotImplementedError

    def H(self):
        """ Return inverse covariance matrix."""
        return self._H

class TDBPhase(object):
    """ Just a mapping between TDBs and phases, for use in bootstrapping
    a model.
    """

    def __init__(self,all_tdbs,all_phases):
        self.tdbs = all_tdbs
        self.phases = all_phases

    def get(self):
        return self.tdbs,self.phases

    def get_poly_fit(self,t0,mask=None,forward=True,dt=100,deg=2):
        ts = self.tdbs[mask]
        ph = self.phases[mask]
        dts = ts - t0
        if forward:
            local_mask = (dts <= 0) & (dts > -dt)
        else:
            local_mask = (dts >= 0) & (dts < dt)
        x = (dts[local_mask]*86400).astype(float)
        y = ph[local_mask].astype(float)
        if np.all(y==0):
            raise ValueError('what')
        p = np.polyfit(x,y,deg)
        return p

class TMParameters(object):
    """Timing model parameters.

    TODO -- Offset handled in here, but what to do in general?
    """

    def __init__(self,m,incoffset=True,tn_func=None):
        self.model = m
        #self.params = [par for par in m.params if not getattr(m,par).frozen]
        self._incoffset = incoffset
        self._offset_val = 0
        self._qsd_only = None
        self._tn_func = tn_func

    def set_tn_func(self,tn_func):
        self._tn_func = tn_func

    def params(self):
        m = self.model
        return [par for par in m.params if not getattr(m,par).frozen]

    def copy(self):
        m = TMParameters(deepcopy(self.model),incoffset=self._incoffset)
        m._offset_val = self._offset_val
        return m

    def set_qsd_only(self,max_fn=1):
        """ max_fn: -1: no spindown, 0 = F0, 1 = F0+F1, etc.
        """
        if self._qsd_only is not None:
            return
        valid_params = ['F%d'%i for i in range(0,max_fn+1)]
        ostate = dict()
        mqsd = self.model
        for param in mqsd.params:
            ostate[param] = getattr(mqsd,param).frozen
            if param not in valid_params:
                getattr(mqsd,param).frozen = True
        self._qsd_only = ostate

    def unset_qsd_only(self):
        if self._qsd_only is None:
            return
        mqsd = self.model
        ostate = self._qsd_only
        for param in mqsd.params:
            getattr(mqsd,param).frozen = ostate[param]
        self._qsd_only = None

    def set_offset(self,offset):
        self._offset_val = offset

    def update_offset(self,delta_offset):
        self._offset_val += delta_offset

    def get_params(self):
        if self._incoffset:
            return ['Offset'] + self.params()
        return self.params()

    def get_values(self):
        p = np.asarray([getattr(self.model,p).value for p in self.params()])
        if self._incoffset:
            p = np.append(self._offset_val,p)
        return p

    def set_values(self,p):
        if self._incoffset:
            self._offset_val = p[0]
            p = p[1:]
        for k,v in zip(self.params(),p):
            getattr(self.model,k).value = v

    def update_values(self,p):
        if self._incoffset:
            self._offset_val += p[0]
            p = p[1:]
        for k,v in zip(self.params(),p):
            getattr(self.model,k).value += v

    def set_value(self,key,val):
        if key == 'Offset':
            self._offset_val = val
        else:
            getattr(self.model,key).value = val

    def phase(self,grid_ts):
        return self.eval(grid_ts)

    def modphase(self,grid_ts):
        return self.modeval(grid_ts)

    def modeval(self,grid_ts):
        ph = self.model.phase(grid_ts,abs_phase=True).frac.value
        if self._incoffset:
            ph -= self._offset_val
        if self._tn_func is not None:
            ph += self._tn_func(
                    get_tdbs(grid_ts,self.model))*self.model.F0.value
        return ph.astype(float)

    def eval(self,grid_ts):
        ph = self.model.phase(grid_ts,abs_phase=True)
        ph = (ph[0]+ph[1]).value
        if self._incoffset:
            ph -= self._offset_val
        if self._tn_func is not None:
            ph += self._tn_func(
                    get_tdbs(grid_ts,self.model))*self.model.F0.value
        return ph

    def designmatrix(self,grid_ts):
        # NB 12/17/2020, PINT interface changed and returns designmatrix
        # in units of time by default, so must scale by F0...  annoying
        D,params,units = self.model.designmatrix(
                grid_ts,incoffset=self._incoffset)
        D *= -self.model.F0.value # scale and change sign convention
        return D.transpose()

    def get_fixed(self,grid_ts):
        m = TMParametersFixed(deepcopy(self.model),grid_ts,
                incoffset=self._incoffset,tn_func=self._tn_func)
        m._offset_val = self._offset_val
        return m

class TMParametersFixed(TMParameters):
    """Timing model parameters.

    TODO -- Offset handled in here, but what to do in general?

    This version stores a frozen TN realization that is included when calling
    phase.
    """

    def __init__(self,m,ts,incoffset=True,tn_func=None):
        super().__init__(m,incoffset=incoffset,tn_func=tn_func)
        self._ts = ts
        if tn_func is not None:
            self._tn = self._tn_func(
                    get_tdbs(self._ts,self.model))*self.model.F0.value
        else:
            self._tn = None

    def copy(self,deepcopy_TOAs=True):
        # TODO -- sort out pint.TOAs copying -- necessary?
        m = deepcopy(self.model)
        if deepcopy_TOAs:
            m = TMParametersFixed(m,deepcopy(self._ts),
                    incoffset=self._incoffset,tn_func=self._tn_func)
        else:
            m = TMParametersFixed(m,self._ts,incoffset=self._incoffset,
                    tn_func=self._tn_func)
        m._offset_val = self._offset_val
        return m

    def phase(self):
        ph = self.model.phase(self._ts,abs_phase=True)
        ph = (ph[0]+ph[1]).value
        if self._incoffset:
            ph -= self._offset_val
        if self._tn is not None:
            ph += self._tn
        return ph

    def modphase(self):
        return self.phase()%1

    def designmatrix(self,float64=True):
        # NB 12/17/2020, PINT interface changed and returns designmatrix
        # in units of time by default, so must scale by F0...  annoying
        D,params,units = self.model.designmatrix(
                self._ts,incoffset=self._incoffset)
        D *= -self.model.F0.value # scale and change sign convention
        return D.transpose()

class TNAmps(object):
    """ NB to future self: the coeffs/params are PSD values, and "scale"
        should be provided to convert PSD to the realization in the data.

        Notionally, scale = (2/TOBS)**0.5*F0.
    """

    def __init__(self,coeffs,freqs,epoch,scale=1):
        """ Frequencies are in 1/day."""
        self.coeffs = coeffs
        self.freqs = freqs
        self.epoch = epoch
        self.scale = scale

    def eval(self,mjds):
        """ NB -- mjds *must* be at the barycenter, i.e including Roemer.
        """
        rvals = np.zeros(len(mjds))
        mjds = mjds - self.epoch
        for ic,(c,s) in enumerate(zip(
                self.coeffs[0::2]*self.scale,
                self.coeffs[1::2]*self.scale)):
            ph = (2*np.pi*self.freqs[ic])*mjds
            rvals += c*np.cos(ph)
            rvals += s*np.sin(ph)
        return rvals

    def modeval(self,mjds):
        return (self.eval(mjds)%1).astype(float)

    def get_psd(self):
        """ Return the coefficients as a PSD, not a data realization."""
        return self.freqs*365,self.coeffs[::2]**2 + self.coeffs[1::2]**2
    
    def get_nharm(self):
        return len(self.freqs)

    def add_harms(self,nharm):
        """ Add (or subtract) additional harmonics.
        """
        curr_nharm = len(self.freqs)
        if nharm < 0:
            new_nharm = max(0,curr_nharm + nharm)
            self.freqs = self.freqs[:new_nharm]
            self.coeffs = self.coeffs[:2*new_nharm]
        else:
            f0 = self.freqs[0]
            self.freqs = np.append(self.freqs,
                    f0*np.arange(curr_nharm+1,curr_nharm+1+nharm))
            self.coeffs = np.append(self.coeffs,np.zeros(2*nharm))

    def get_values(self):
        return self.coeffs

    def update_coeffs(self,dc):
        self.coeffs += dc

    def update_values(self,dc):
        self.coeffs += dc

    def set_coeffs(self,c):
        self.coeffs = np.asarray(c).copy()

    def set_values(self,c):
        self.set_coeffs(c)

    def copy(self):
        return TNAmps(self.coeffs.copy(),self.freqs.copy(),
                self.epoch,self.scale)

    def designmatrix(self,mjds):
        ndata = len(mjds)
        nharm = len(self.freqs)
        F = np.empty([2*nharm,ndata],dtype=np.float128)
        phase = np.empty_like(mjds)
        times = mjds-self.epoch
        for iharm,freq in enumerate(self.freqs):
            phase[:] = (2*np.pi*freq)*times
            np.cos(phase,out=F[2*iharm,:])
            np.sin(phase,out=F[2*iharm+1,:])
        F *= self.scale
        return F

    def get_fixed(self,mjds):
        return TNAmpsFixed(self.coeffs.copy(),self.freqs.copy(),
                self.epoch,mjds,scale=self.scale)

    def get_params(self):
        rvals = deque()
        for i in range(len(self.coeffs)//2):
            rvals.append('COS%d'%i)
            rvals.append('SIN%d'%i)
        return list(rvals)

    def as_parfile(self,f0_scale):
        if len(self.freqs) == 0:
            return ''
        wave_om = 2*np.pi*self.freqs[0]
        wave_epoch = self.epoch
        s = 'WAVEEPOCH %.15f\n'%(wave_epoch)
        s += 'WAVE_OM %.15f 0\n'%(wave_om)
        vals = self.coeffs * self.scale / f0_scale
        cos_vals = vals[::2]
        sin_vals = vals[1::2]
        for ival in range(len(cos_vals)):
            s += 'WAVE%d %.15f %.15f\n'%(
                    ival+1,sin_vals[ival],cos_vals[ival])
        return s

    def as_pint_component(self,f0_scale):
        """ Return an equivalent WAVE model that can be added to a PINT
        timing model (model.add_component).
        """
        wave_om = 2*np.pi*self.freqs[0]
        wave_epoch = self.epoch
        s = 'WAVEEPOCH %.15f\n'%(wave_epoch)
        s += 'WAVE_OM %.15f 0\n'%(wave_om)
        vals = self.coeffs * self.scale / f0_scale
        cos_vals = vals[::2]*u.s
        sin_vals = vals[1::2]*u.s
        m = wave.Wave()
        m.WAVE_OM.quantity = wave_om / u.day
        m.WAVEEPOCH. value = wave_epoch
        m.WAVE1.quantity = [sin_vals[0],cos_vals[0]]
        for i in range(1,len(cos_vals)):
            name = 'WAVE%d'%(i+1)
            p = prefixParameter(name=name,units='s',type_match='pair',
                    long_double=True,parameter_type='pair')
            p.quantity = [sin_vals[i],cos_vals[i]]
            m.add_param(p)
        return m

class TNAmpsFixed(TNAmps):
    """ NB to future self: the coeffs/params are PSD values, and "scale"
        should be provided to convert PSD to the realization in the data.

        Notionally, scale = (2/TOBS)**0.5*F0.

        This version uses a static set of MJDs to pre-evaluate quantities.
    """

    def __init__(self,coeffs,freqs,epoch,mjds,scale=1):
        """ Frequencies are in 1/day."""
        super().__init__(coeffs,freqs,epoch,scale=scale)
        self.update_mjds(mjds)

    def phase(self):
        #return np.einsum('ij,j',self._Ft,self.coeffs)
        return np.inner(self._Ft,self.coeffs)

    def modphase(self):
        return self.phase()

    def copy(self):
        return TNAmpsFixed(self.coeffs.copy(),self.freqs.copy(),
                self.epoch,self.mjds,scale=self.scale)

    def designmatrix(self,float64=True):
        # NB float64 kwarg doesn't do anything yet
        return self._F

    def update_mjds(self,mjds):
        self.mjds = np.asarray(mjds).copy()
        self._F = super().designmatrix(mjds) 
        self._Ft = self._F.transpose().copy()

    def add_harms(self,nharm):
        super().add_harms(nharm)
        self.update_mjds(self.mjds)

class TMTNJoint(object):

    def __init__(self,tmp,tna):
        """ tmp == instance of TMParameters_Fermi
            tna == instance of TNAmpsFixed
        """
        self._tmp = tmp
        self._tna = tna

    def tm(self):
        return self._tmp

    def tn(self):
        return self._tna

    def pint_model(self):
        return self._tmp.model

    def pm(self):
        return self._tmp.model

    def get_ntm_param(self):
        return len(self._tmp.get_values())

    def get_nharm(self):
        return len(self.tn().freqs)

    def get_allow_f2(self):
        m = self.pint_model()
        comp,order,comp_type_list,comp_type = m.map_component('Spindown')
        # if we had an original model, there could be some kruft with old
        # FN terms hanging around.  But the num_spin_terms should be set
        # correctly, so checking that too will yield a good test
        c1 = comp.num_spin_terms >= 3
        c2 = hasattr(m,'F2') and (not m.F2.frozen)
        return c1 and c2

    def get_max_fn(self):
        m = self.pint_model()
        comp,order,comp_type_list,comp_type = m.map_component('Spindown')
        # if we had an original model, there could be some kruft with old
        # FN terms hanging around.  But the num_spin_terms should be set
        # correctly, so checking that too will yield a good test
        max_fn = 1
        if not comp.num_spin_terms >= 3:
            return max_fn
        next_fn = 2
        while True:
            attr = 'F%d'%next_fn
            if hasattr(m,attr) and (not getattr(m,attr).frozen):
                max_fn += 1
                next_fn += 1
            else:
                break
        return max_fn

    def get_htn(self,p,zeropad=True,ntm_param=None):
        ls_freqs = self.tn().freqs*365.25
        # this is stupid, but this call is stupidly slow, so provide an
        # alternative way to furnish the information
        if ntm_param is None:
            ntm_param = self.get_ntm_param()
        H_tn = TNMatrix(p,ls_freqs,zeropad=ntm_param if zeropad else 0).H()
        return H_tn

    def copy(self,deepcopy_TOAs=True):
        return TMTNJoint(self._tmp.copy(deepcopy_TOAs=deepcopy_TOAs),self._tna.copy())

    def get_values(self):
        return np.append(self._tmp.get_values(),self._tna.get_values())

    def set_values(self,v):
        np1 = len(self.tm().get_params())
        self._tmp.set_values(v[:np1])
        self._tna.set_values(v[np1:])

    def update_values(self,v):
        np1 = len(self.tm().get_params())
        self._tmp.update_values(v[:np1])
        self._tna.update_values(v[np1:])

    # TODO -- super inefficient memory-wise, especially with 1e5s of phs
    def designmatrix(self,float64=True):
        D = self._tmp.designmatrix()
        F = self._tna.designmatrix()
        dtype = np.float64 if float64 else D.dtype
        output = np.empty((D.shape[0]+F.shape[0],D.shape[1]),dtype=dtype)
        output[:D.shape[0]] = D
        output[D.shape[0]:] = F
        return output
        #return np.matrix(np.append(D,F,axis=0))

    def phase(self,grid_ts=None):
        if grid_ts is not None:
            grid_tdbs = get_tdbs(grid_ts,self._tmp.model)
            return self._tmp.eval(grid_ts) + self._tna.eval(grid_tdbs)
        return self._tmp.phase() + self._tna.phase()

    def modphase(self,grid_ts=None):
        # TODO -- throw away integer parts earlier for greater precision
        # have written most of the child methods already
        ph = self.phase(grid_ts=grid_ts)
        ph %= 1
        return ph.astype(float)

    def get_psd(self):
        return self._tna.get_psd()

    # OBSOLETE?
    def refine_coeffs(self,lct,weights,H_tn=None,niter=4,ftol=None,
            stabilize=0):
        we = weights
        logls = []
        for i in range(niter):
            ph,fph,M,G,H = self.template_goodies(lct,we,stabilize=stabilize)
            logls.append(np.sum(np.log(we*fph+(1-we))))
            print(logls)
            if H_tn is not None:
                print('adding TN')
                G = G - np.inner(H_tn,self.get_values()) # lambda*H_tn
                H = H + H_tn
            c = cho_factor(H,lower=True)
            coeffs = cho_solve(c,G)
            self.update_values(coeffs)
            if (ftol is not None) and (len(logls)>1):
                if abs(logls[-1]-logls[-2])<ftol:
                    break
        fph = lct(self.phase()%1)
        logls.append(np.sum(np.log(we*fph+(1-we))))
        return np.asarray(logls)

    def get_subspace_G(self,G):
        #ntm_param = self.get_ntm_param()
        ntn_param = 2*self.get_nharm()
        ntm_param = G.shape[0]-ntn_param
        G_d = G[:ntm_param]
        G_f = G[ntm_param:]
        return G_d,G_f

    def get_subspace_H(self,H):
        #ntm_param = self.get_ntm_param()
        ntn_param = 2*self.get_nharm()
        ntm_param = H.shape[0]-ntn_param
        H_df = H[:ntm_param,ntm_param:] 
        H_dd = H[:ntm_param,:ntm_param:]
        H_ff = H[ntm_param:,ntm_param:]
        return H_dd,H_df,H_ff

    def get_params(self):
        return self.tm().get_params() + self.tn().get_params()

    def get_tm_params(self):
        return self.tm().get_params()

    def as_parfile(self,start=None,finish=None,p_tn=None,p_tne=None,
            additional_lines=None):
        tm = self.tm()
        s1 = tm.model.as_parfile()
        f0_scale = tm.model.F0.value
        s2 = self.tn().as_parfile(f0_scale)
        output = s1 + s2
        lines = output.split('\n')
        replace = utils.lines_replace_value
        start = start or (self.tn().mjds[0]-0.001)
        finish = finish or (self.tn().mjds[-1]+0.001)
        lines = replace(lines,'START','%.5f'%start)
        lines = replace(lines,'FINISH','%.5f'%finish)
        lines = replace(lines,'UNITS','TDB')

        # It's not really possible to have a completely consistent
        # reference phase due to the way we shuffle timing noise back and
        # forth between PINT and this code.  Therefore, simply write out
        # a quantity that can be read back in to set the phase as needed.
        lines = replace(lines,'LATPHASE','%.10f'%(tm._offset_val%1))

        if p_tn is not None:
            # now add the LAT timing noise
            if p_tne is None:
                lines = replace(lines,'LATAMP','%.5f'%(p_tn[0]))
                lines = replace(lines,'LATIDX','%.5f'%(p_tn[1]))
            else:
                lines = replace(lines,'LATAMP','%.5f %.2f'%(p_tn[0],p_tne[0]))
                lines = replace(lines,'LATIDX','%.5f %.2f'%(p_tn[1],p_tne[1]))
            lines = replace(lines,'LATNHARM','%d'%(len(self.tn().freqs)))

        if additional_lines is not None:
            lines += list(additional_lines)
        return '\n'.join(lines) + '\n'

    def change_timerange(self,new_ts):
        """ Return a copy of the model with a different dataset.
        NB this does *not* change the underlying frequencies of the timing
        noise modes.  So it's good for extrapolation but not necessarily
        re-fitting.
        """
        return self.get_fixed(new_ts)

    def get_fixed(self,new_ts):
        tn_new = self.tn().get_fixed(get_tdbs(new_ts,self.tm().model))
        tm_new = self.tm().get_fixed(new_ts)
        return TMTNJoint(tm_new,tn_new)

    def extrap_F0_F1(self,t0,forward=True,dt=None):
        """ Compute phase, frequency, and frequency derivative (bary) for
        the end of the data set, or the beginning, depending on "forward".
        """
        tdbs = self.tn().mjds
        phas = self.phase()
        idx0 = np.searchsorted(tdbs,t0)
        if idx0 == len(tdbs):
            idx0 -= 1
        if forward:
            if dt is None:
                idx0 = max(3,idx0)
                x = tdbs[idx0-3:idx0]
                y = phas[idx0-3:idx0]
            else:
                ix = np.searchsorted(tdbs,tdbs[idx0]-dt)
                x = tdbs[ix:idx0]
                y = phas[ix:idx0]
            phi0 = phas[-1]
            epoch = x[-1]
        else:
            if dt is None:
                idx0 = min(len(tdbs)-3,idx0)
                x = tdbs[idx0:idx0+3]
                y = phas[idx0:idx0+3]
            else:
                ix = np.searchsorted(tdbs,tdbs[idx0]+dt)
                x = tdbs[idx0:ix]
                y = phas[idx0:ix]
            phi0 = phas[0]
            epoch = x[0]
        x = (x-epoch)*86400
        p = np.polyfit(x.astype(float),y.astype(float),2)
        F1,F0,irrelevant_offset = p
        # NB use Taylor convention rather than polynomial expansion
        F1 *= 2
        return epoch,phi0,F0,F1

    def get_typical_F0_F1(self,timescale,n=100,perc=95):
        """ Use n random locations of length timescale (d) to estimate the
        typical effective F0/F1 of the timing noise."""
        t0 = self.tn().mjds[0]
        t1 = self.tn().mjds[-1]
        # ensure there are at least 4 data points per timescale to avoid
        # problems with conditioning
        nmjd = int((t1-t0)/(timescale*0.25))
        mjds = np.linspace(self.tn().mjds[0],self.tn().mjds[-1],nmjd).astype(float)
        tn = self.tn().eval(mjds).astype(float)
        max_index = np.searchsorted(mjds,mjds[-1]-timescale)
        if max_index/len(mjds) < 0.1:
            print('Warning, requesting F0/F1 variations over timescales long compared to length of dataset.')
        indices = (np.random.rand(n)*max_index).astype(int)
        ps = np.empty([n,3])
        devs = np.empty(n)
        for i in range(n):
            i0 = indices[i]
            i1 = np.searchsorted(mjds,mjds[i0]+timescale)
            local_mjds = mjds[i0:i1]
            dt = (local_mjds-local_mjds[0])*86400
            try:
                ps[i] = np.polyfit(dt,tn[i0:i1],2)
                devs[i] = np.std(tn[i0:i1]-np.polyval(ps[i],dt))
            except ValueError:
                print(i0,i1,mjds[i0],mjds[i1])
        F0_scale = np.percentile(np.abs(ps[:,1]),[perc])[0]
        F1_scale = np.percentile(np.abs(ps[:,0]),[perc])[0]
        dev_scale = np.percentile(devs,[perc])[0]
        return F0_scale,F1_scale,F0_scale*timescale*86400,F1_scale*(timescale*86400)**2,dev_scale

    def get_typical_F0_F1_2(self,dt1,dt2,n=100,perc=95,F0_only=False):
        """ Fit a set of data of length dt1 with F0/F1, then add the the
        subsequent length of data, dt2, and fit it.  Examine the difference
        in F0/F1 resulting."""
        t0 = self.tn().mjds[0]
        t1 = self.tn().mjds[-1]
        # ensure there are at least 4 data points per timescale to avoid
        # problems with conditioning
        nmjd = int((t1-t0)/(min(dt1,dt2)*0.25))
        mjds = np.linspace(t0,t1,nmjd).astype(float)
        tn = self.tn().eval(mjds).astype(float)
        max_index = np.searchsorted(mjds,mjds[-1]-(dt1+dt2))
        if (dt1+dt2)/(t1-t0) > 0.2:
            print('Warning, requesting F0/F1 variations over timescales long compared to length of dataset, so there are few realizations.')
        indices = (np.random.rand(n)*max_index).astype(int)
        ps = np.empty([n,3-int(F0_only)])
        devs = np.empty(n)
        for i in range(n):
            i0 = indices[i]
            i1 = np.searchsorted(mjds,mjds[i0]+dt1)
            dt = (mjds[i0:i1]-mjds[i0])*86400
            p1 = np.polyfit(dt,tn[i0:i1],2-int(F0_only))
            i2 = np.searchsorted(mjds,mjds[i0]+dt1+dt2)
            dt = (mjds[i0:i2]-mjds[i0])*86400
            y2 = tn[i0:i2]-np.polyval(p1,dt)
            #p2 = np.polyfit(dt,tn[i0:i2],2)
            p2 = np.polyfit(dt,y2,2-int(F0_only))
            devs[i] = np.std(y2-np.polyval(p2,dt))
            #print(np.std(tn[i0:i2]-np.polyval(p2,dt)))
            #print(p1,p2)
            #ps[i] = p2-p1
            ps[i] = p2
        if F0_only:
            F0_scale = np.percentile(np.abs(ps[:,0]),[perc])[0]
            F1_scale = 0
        else:
            F0_scale = np.percentile(np.abs(ps[:,1]),[perc])[0]
            F1_scale = np.percentile(np.abs(ps[:,0]),[perc])[0]
        return F0_scale,F1_scale,F0_scale*(dt1+dt2)*86400,F1_scale*((dt1+dt2)*86400)**2,np.percentile(devs,[perc])[0]

    def get_tn_fixed(self):
        """ Return a TMParametersFixed object including the timing noise
        component implemented via the "tn_func" interface.  This provides
        a fittable object with frozen timing noise.
        """
        tmp = TMParametersFixed(self.pm(),self.tm()._ts,
                tn_func = lambda x: self.tn().phase()/self.pm().F0.value)
        tmp.set_values(self.tm().get_values())
        return tmp


def clean_model(morig,grid_ts,max_fn=1,center_epoch=True):
    """ Given a timing model, perhaps with >=F2, WAVES, or IFUNCs, make a
        new model and delete all that nonsense.  Return the new model along
        with the difference in phase evaluated at the provided grid TOAs.
    """
    mclean = deepcopy(morig)
    ph_orig = mclean.phase(grid_ts,abs_phase=True)
    if center_epoch:
        mjds = grid_ts.get_mjds().value
        new_epoch = 0.5 * (mjds.max() + mjds.min())
        mclean.PEPOCH.value = new_epoch
    utils.remove_taylor(mclean,max_f=max_fn)
    
    # make sure F0 and F1 are in the model and free
    utils.add_spindown_component(mclean,degree=0,free=True)
    utils.add_spindown_component(mclean,degree=1,free=True)
    for i in range(2,max_fn+1):
        utils.add_spindown_component(mclean,degree=i,free=True)
    utils.remove_waves(mclean,no_wave=True)
    utils.remove_ifuncs(mclean)
    utils.remove_rococo_glitches(mclean)
    ph_post = mclean.phase(grid_ts,abs_phase=True)
    dphi = ph_orig-ph_post
    return mclean,dphi.int.value + dphi.frac.value

def fit_model(mjoint,target_phase,quiet=False,eff_wn=0.01,H_tn=None,
        niter=2):
    """ Fit the provided model using the target phase.

    The effective white noise only comes in to play if a timing noise
    constraint matrix is also provided, in which case it will help balance
    the goodness-of-fit with the spectral coefficients.
    """

    for i in range(niter):
        resid = target_phase - mjoint.phase()
        #M = np.matrix(mjoint.designmatrix())
        M = mjoint.designmatrix()
        if eff_wn is not None:
            Hw = np.diag(np.ones(len(resid))/eff_wn**2)
            T = Hw@M.T
            H = M@T
            X = resid@T
        else:
            H = M@M.T
            X = resid@M.T
        if H_tn is not None:
            H += H_tn
            X -= np.inner(H_tn,mjoint.get_values())

        # 2/5/2021 -- based on test results, the big problem here is
        # parameter scaling, so apply an ad hoc scale to improve
        # the conditioning!
        scale = np.diag(H)**-0.5
        Hscale = H*np.outer(scale,scale)
        Xscale = X*scale

        # switch to least squares for more robust solution?
        try:
            #c = cho_factor(H,lower=True)
            c = cho_factor(Hscale,lower=True)
        except LinAlgError:
            print('failed to invert initial hessian')
            #H += np.diag(np.diag(H)*1e-3)
            #c = cho_factor(H,lower=True)
            Hscale += np.diag(np.diag(Hscale)*1e-3)
            c = cho_factor(Hscale,lower=True)
        #coeffs = cho_solve(c,X)
        coeffs = cho_solve(c,Xscale)*scale
        """
        # TMP
        # use this to study conditioning of matrices etc.
        debug_locals.update(locals())
        import sys; sys.exit(0)
        # end TMP
        """
        """
        # rcond=None uses machine*epsilon times largest dimension
        print(M.shape,H.shape,X.shape)
        coeffs,resids,rank,singvals = np.linalg.lstsq(
                (H).astype(float),(X).astype(float),rcond=None)
        print(H.shape,X.shape,coeffs.shape,rank)
        """
        mjoint.update_values(coeffs)

        # post facto, for some reason the offset isn't well determined,
        # so take out the median
        # TODO 2/5/2021 -- this might work better now with conditioning,
        # maybe can be removed.
        resid = target_phase - mjoint.phase()
        mjoint.tm()._offset_val -= np.median(resid)
        #mjoint.tm()._offset_val %= 1


    # make sure the model is good enough to proceed.  If it isn't, the TN
    # may be too strong for the selected number of harmonics
    resid = target_phase - mjoint.phase()
    if np.max(np.abs(resid)) > 0.05:
        if not quiet:
            print('Warning!  The deviation (max=%.2f, std=%.2f) in phase between the original and the resulting model is possibly too large to result in good least squares fits.  You may need to use more harmonics for the level of timing noise.'%(np.max(np.abs(resid)),np.std(resid)))
        return False
    return True

def fit_initial_model(mjoint,target_phase,max_fn=1,quiet=False,
        p_tn=None,niter=4):
    """ Return a good model for fitting timing noise.
    This simply finds good estimates of parameters and TN coeffs using
    "the truth" from the known model used to simulate the phases and
    the subsequent timing noise added.  This gets us to the state
    we would normally be, i.e. having a decent timing solution to
    start from.

    Because we are not using timing noise, only do a QSD fit here, no
    other free parameters.  For modestly steep spectra, this will
    result in a relatively unbiased set of TN coeffs.  More importantly,
    it won't result in janky positions.
    """
    mjoint.tm().set_qsd_only(max_fn=max_fn)
    H_tn = None
    if p_tn is not None:
        H_tn = mjoint.get_htn(p_tn,zeropad=True)
    # This code wasn't effecitve, setting maxloop to one.  More effective
    # to remove the offset and add harmonics as necessary.
    maxloop = 1
    for i in range(maxloop):
        if i < maxloop-1:
            myquiet = True
        else:
            myquiet = quiet
        fit_ok = fit_model(mjoint,target_phase,quiet=myquiet,H_tn=H_tn,
                niter=niter)
        if not fit_ok:
            niter += 3
        else:
            break
    mjoint.tm().unset_qsd_only()
    return fit_ok

class DataSet(object):
    """ General object to match a timing solution and data together.

    Children specific to photons or TOAs implement various "virtual"
    methods.
    """

    def __init__(self,par,ts,tn_func=None,**model_kwargs):
        """ par -- PINT timing model
            ts -- PINT TOAs object
            tn_func -- return TN(s) as a function of Time(TDB); if given
                will apply this to any generated TMParams objects such that
                the data will behave as if a TN signal is present.
        """
        self._ts = ts
        self._model_kwargs = model_kwargs
        self._tn_func = tn_func
        #print('DataSet set_par')
        self.set_par(par)

    def __getstate__(self):
        state = self.__dict__.copy() # shallow
        state.pop('_morig')
        return state

    def __setstate__(self,state):
        self.__dict__.update(state)
        # NB -- we have to decide now whether to honor the pickled state of
        # the dataset, or if the ephemeris has changed on disk whether to
        # udpate the parameters.  Don't know what's best, but for now, am
        # opting to re-load the ephemeris
        #print('setstate set_par')
        self.set_par(self._par,quiet=True)
        #self._morig = utils.get_model(self._par)


    def set_timerange(self,minmjd=None,maxmjd=None):
        if minmjd is not None:
            self._t0 = minmjd
        if maxmjd is not None:
            self._t1 = maxmjd
        self.tobs = (self._t1-self._t0)/365.25
        if hasattr(self._ts,'table_selects'):
            nstack = len(self._ts.table_selects)
            for i in range(nstack):
                self._ts.unselect()
        mjds = self._ts.get_mjds().value
        # TODO -- assume TOAs are sorted, but I don't think this is
        # guaranteed anywhere
        # TODO -- cache this full data set after first call
        self._ur_t0 = mjds[0]
        self._ur_t1 = mjds[-1]
        i0 = np.searchsorted(mjds,self._t0,side='left')
        i1 = np.searchsorted(mjds,self._t1,side='right')
        mask = np.zeros(len(mjds),dtype=bool)
        mask[i0:i1] = True
        alt_mask = (mjds >= self._t0) & (mjds <= self._t1)
        # TMP!  sanity check
        self._select_i0 = i0
        self._select_i1 = i1
        assert(np.all(alt_mask==mask))
        self._ts.select(mask)
        self._select_mask = mask

    def clear_timerange(self,slop=15):
        if hasattr(self._ts,'table_selects'):
            nstack = len(self._ts.table_selects)
            for i in range(nstack):
                self._ts.unselect()
        mjds = self._ts.get_mjds().value
        self.set_timerange(minmjd=mjds[0]-slop,maxmjd=mjds[-1]+slop)

    def calculate_timestep(self,dweight,forward=True,gap_thresh_frac=0.6,
            gap_thresh_dur=0.4):
        """ Determine the smallest possible boundaries increasing the total
        photon weights by dweight.

        forward -- expand boundaries forward in time; else, backwards
            if boundary is already at the end of the data, return 0
        """
        old_t0 = self._t0
        old_t1 = self._t1
        new_t0,new_t1 = old_t0,old_t1
        self.clear_timerange()
        mjds = self._ts.get_mjds().value

        if forward and (old_t1 < self._t1):
            idx0 = np.searchsorted(mjds,old_t1)
            W = np.cumsum(self.get_weights()[idx0:])
            idxW = np.searchsorted(W,dweight)
            assert((idx0+idxW) < len(mjds))
            new_t1 = mjds[idx0+idxW]

            local_mjds = mjds[idx0:idx0+idxW+1]
            gaps = local_mjds[1:]-local_mjds[:-1]
            if gaps.max()/gaps.sum() > gap_thresh_dur:
                # there is a sufficiently long gap; if it is towards the
                # end of the interval, terminate the timestep early
                agap = np.argmax(gaps)
                if W[agap]/W[idxW] > gap_thresh_frac:
                    print('Return expanded timestep going well past gap.')
                    #new_t1 = mjds[idx0+agap]
                    self.set_timerange(old_t0,old_t1)
                    print('dweight_frac = ',W[agap]/W[idxW]*2)
                    return self.calculate_timestep(
                            dweight*W[agap]/W[idxW]*2,forward=True,
                            gap_thresh_frac=1.0,gap_thresh_dur=1.0)


        if (not forward) and (old_t0 > self._t0):
            raise NotImplementedError('Need to code up gap logic.')
            idx0 = np.searchsorted(mjds,old_t0)-1
            W = np.cumsum(self.get_weights()[idx0:][::-1])
            idxW = np.searchsorted(W,dweight)
            assert((idx0-idxW)>0)
            new_t0 = mjds[idx0-idxW]

        self.set_timerange(old_t0,old_t1)
        return new_t0,new_t1

    def calculate_timestep2(self,dweight,t0,gap_thresh_frac=0.6,
            gap_thresh_dur=0.4):
        """ dweight: amount of weights to accumulate
            t0: starting time (MJD)
        """

        old_t0 = self._t0
        old_t1 = self._t1
        self.clear_timerange()
        mjds = self._ts.get_mjds().value

        if (t0 >= mjds[-1]) or (t0 <= mjds[0]):
            self.set_timerange(old_t0,old_t1)
            return [None]*6

        # compute the forward step
        idx0 = np.searchsorted(mjds,t0)
        W = np.cumsum(self.get_weights()[idx0:])
        idxW = np.searchsorted(W,dweight)
        idx_end = min(len(mjds)-1,idx0 + idxW)
        idxW = min(idxW,len(W)-1)

        # detect any gaps
        local_mjds = mjds[idx0:idx_end+1]
        gaps = local_mjds[1:]-local_mjds[:-1]
        if gaps.max()/gaps.sum() > gap_thresh_dur:
            # there is a sufficiently long gap; if it is towards the
            # end of the interval, terminate the timestep early
                agap = np.argmax(gaps)
                if W[agap]/W[idxW] > gap_thresh_frac:
                    print('shortening forward step')
                    idx_end = idx0 + agap
                    #dweight_factor = W[agap]/W[idxW]

        # now compute the backward step
        W = np.cumsum(self.get_weights()[idx0:][::-1])
        idxW = np.searchsorted(W,dweight)
        idx_beg = idx0 - idxW
        idx_beg = max(0,idx0 - idxW)
        idxW = min(idxW,len(W)-1)

        # detect any gaps
        local_mjds = mjds[idx_beg:idx0]
        gaps = local_mjds[1:]-local_mjds[:-1]
        if gaps.max()/gaps.sum() > gap_thresh_dur:
            # there is a sufficiently long gap; if it is towards the
            # end of the interval, terminate the timestep early
                agap = np.argmax(gaps)
                if W[agap]/W[idxW] > gap_thresh_frac:
                    idx_beg = idx0 - agap
                    print('shortening backward step')
                    

        tbeg = mjds[idx_beg]
        tend = mjds[idx_end]
        forward_mask = np.zeros(len(mjds),dtype=bool)
        forward_mask[idx0:idx_end+1] = True
        backward_mask = np.zeros(len(mjds),dtype=bool)
        backward_mask[idx_beg:idx0] = True
        Wf = np.sum(self.get_weights()[forward_mask])
        Wb = np.sum(self.get_weights()[backward_mask])
        self.set_timerange(old_t0,old_t1)
        return tbeg,tend,forward_mask,backward_mask,Wf,Wb

    def calculate_timesteps(self,dweight):
        """ Determine the smallest possible boundaries increasing the total
        photon weights by dweight.

        Return entire set out to the edge of the data, both directions.
        """
        old_t0 = self._t0
        old_t1 = self._t1
        self.clear_timerange()
        mjds = self._ts.get_mjds().value

        # forward
        idx1 = np.searchsorted(mjds,old_t1)
        W1 = np.cumsum(self.get_weights()[idx1:])
        nforward = int(W1[-1]/dweight)+1
        idx_forward = np.searchsorted(W1,dweight*np.arange(1,nforward+1))
        idx_forward[-1] -= 1
        forward_steps = mjds[idx1:][idx_forward]
        # backward
        idx0 = np.searchsorted(mjds,old_t0)
        W0 = np.cumsum(self.get_weights()[:idx0][::-1])
        nbackward = int(W0[-1]/dweight)+1
        idx_backward = np.searchsorted(W0,dweight*np.arange(1,nbackward+1))
        idx_backward[-1] -= 1
        backward_steps = mjds[:idx0][::-1][idx_backward]

        self.set_timerange(old_t0,old_t1)
        return forward_steps,backward_steps

    def apply_toa_mask(self,mask):
        if hasattr(self._ts,'table_selects'):
            nstack = len(self._ts.table_selects)
            for i in range(nstack):
                self._ts.unselect()
        self._ts.select(mask)
        self._select_mask = mask

    def trim_mask(self,mask):
        """ Take a mask applying to a full data set and trim it to match
        the current TOA selection.
        """
        if len(mask) != len(self._select_mask):
            raise ValueError('Masks do not match.')
        #istart = np.searchsorted(self._select_mask,0.1)
        #istop = istart + np.sum(self._select_mask)
        return mask[self._select_i0:self._select_i1]
        #return mask[istart:istop]

    def expand_mask(self,mask):
        """ Take a mask applying to the current data and zero-pad it to
        the full data.
        """
        #istart = np.searchsorted(self._select_mask,0.1)
        #istop = istart + np.sum(self._select_mask)
        istart,istop = self._select_i0,self._select_i1
        a1 = np.append(np.zeros(istart,dtype=bool),mask)
        a2 = np.append(a1,np.zeros(len(self._select_mask)-istop,dtype=bool))
        assert(len(a2)==len(self._select_mask))
        return a2

    def set_tn_func(self,tn_func):
        self._tn_func = tn_func

    def ntoas(self,select=True):
        if select:
            return self._ts.ntoas
        return len(self._select_mask)

    def get_tn_scale(self,add_tobs=0):
        TN_SCALE = ((self.tobs+add_tobs)/2)**-0.5*self.F0_scale
        return TN_SCALE

    def set_par(self,newpar,quiet=False):
        if not quiet:
            print('set_par called with',newpar)
        self._morig = utils.get_pint_model(newpar,**self._model_kwargs)
        self._par = newpar
        self.PEPOCH = self._morig.PEPOCH.value
        # always keep same F0_scale, or...?
        self.F0_scale = self._morig.F0.value

        # set initial bounds based on ephemeris
        t0,t1 = utils.get_bounds(self._par)
        mjds = self._ts.get_mjds().value
        # I don't remember where this 15-day slop comes from.  Trim?
        if t0 is None:
            t0 = np.floor(mjds.min())-15
        else:
            t0 = float(t0)
        if t1 is None:
            t1 = np.ceil(mjds.max())+15
        else:
            t1 = float(t1)

        self.set_timerange(t0,t1)

    def get_freqs(self,nharm,add_tobs=0):
        tobs_eff = self.tobs + add_tobs
        #return np.arange(1,nharm+1)*(1./self.tobs)
        return np.arange(1,nharm+1)*(1./tobs_eff)

    def get_ts(self):
        return self._ts

    def get_mjds(self):
        return self._ts.get_mjds().value

    def get_tdbs(self):
        """ Return photon timestamps in TDB *at the barycenter*."""
        return get_tdbs(self.get_ts(),self._morig)

    def get_grid_ts(self,nsamp):
        tbin = self.tobs/nsamp
        grid_ts,grid_mjds = pint_toas_from_mjds(
                self._t0 + np.arange(nsamp)*tbin*365.25,
                ephem=self._morig.EPHEM.value,site='bary')
        return grid_ts,grid_mjds

    def get_tn(self,nharm,grid_mjds=None,add_tobs=0,waveepoch=None):
        """ NB -- grid_mjds ought to be at the SSB!
        """
        freqs = self.get_freqs(nharm,add_tobs=add_tobs)/365.25
        p0 = np.zeros(2*nharm)
        if waveepoch is None:
            waveepoch = self.PEPOCH
        tn_init = TNAmps(p0,freqs,waveepoch,
                self.get_tn_scale(add_tobs=add_tobs))
        if grid_mjds is not None:
            return tn_init.get_fixed(grid_mjds)
        return tn_init

    def get_pulse_numbers(self):
        return np.asarray(self._ts.get_pulse_numbers())

    def get_measurement_errors(self):
        """ Return measurement errors *in phase*."""
        errs = np.asarray(self._ts.get_errors().to(u.s))
        return errs * self.F0_scale

    def get_tm(self,grid_ts=None,include_tn=False):
        """ Return the initial timing model as a TMParameters object.

        grid_ts -- if provided produce a TNParametersFixed object
        include_tn -- pass along the tn_func
        """
        tm_init = TMParameters(self._morig)
        if include_tn:
            tm_init.set_tn_func(self._tn_func)
        if grid_ts is not None:
            return tm_init.get_fixed(grid_ts)
        return tm_init

    def get_grid_model(self,nharm,ngrid=512,add_tobs=0):
        """ This method is primarily used for getting a model that can be
        tuned up.  E.g., we have added some data and have a predicted phase
        but we need to update the TN coeffs.
        """
        grid_ts,grid_mjds = self.get_grid_ts(512)
        tm_init = self.get_tm(grid_ts=grid_ts)
        tn_init = self.get_tn(nharm,grid_mjds=grid_mjds,add_tobs=add_tobs)
        mjoint_init = TMTNJoint(tm_init,tn_init)
        return grid_ts,mjoint_init

    def get_tn_model(self,nharm,max_fn=1,add_tobs=0,
            center_epoch=True,use_pulse_number=True,quiet=False,
            p_tn=None):
        """ Return a model suitable for a timing noise fit, e.g. by
        removing extraneous spin-down terms from the base model, fitting
        it to the original phase, etc.
        """

        minit,ph_init = clean_model(self._morig,self._ts,
                max_fn=max_fn,center_epoch=center_epoch)
        tn_init = self.get_tn(nharm,grid_mjds=self.get_tdbs(),
                add_tobs=add_tobs,waveepoch=minit.PEPOCH.value)
        tm_init = TMParametersFixed(minit,self._ts)
        mjoint_init = TMTNJoint(tm_init,tn_init)
        if use_pulse_number:
            target_phase = self.get_pulse_numbers()
        else:
            target_phase = mjoint_init.phase() + ph_init
        fit_ok = fit_initial_model(mjoint_init,target_phase,max_fn=max_fn,
                quiet=quiet,p_tn=p_tn,niter=6)
        if not fit_ok:
            raise ValueError('Initial model no good!  Failing.')
        return mjoint_init

    def get_tn_model_exact(self):
        """ Treat the input ephemeris as matching exactly onto our
        expectation and simply make the right TMTNJoint object.

        Primarily intended e.g. when loading up "tnfit" results.
        """
        # strip WAVE from timing model but add it to 
        minit = deepcopy(self._morig)
        comp, order, comp_type_list, comp_type = minit.map_component('Wave')
        comp_type_list.pop(order)

        epoch = comp.WAVEEPOCH.value
        fundamental = comp.WAVE_OM.value/(2*np.pi)
        nharm = comp.num_wave_terms
        freqs = np.arange(1,nharm+1)*fundamental
        # NB -- this is an assumption -- this scale MAY NOT be the same...
        scale = self.get_tn_scale()
        f0_scale = minit.F0.value
        sin_vals = [getattr(comp,'WAVE%d'%i).quantity[0].value for i in range(1,nharm+1)]
        cos_vals = [getattr(comp,'WAVE%d'%i).quantity[1].value for i in range(1,nharm+1)]
        coeffs = np.empty(2*nharm)
        coeffs[::2] = cos_vals
        coeffs[1::2] = sin_vals
        coeffs *= f0_scale / scale

        # special case -- we have enabled an F2 fit but the parameter is
        # missing
        max_fn = get_par_max_fn(self._par)
        for i in range(2,max_fn+1):
            utils.add_spindown_component(minit,degree=i,free=True)

        tn = TNAmpsFixed(coeffs,freqs,epoch,self.get_tdbs(),scale=scale)
        tm = TMParametersFixed(minit,self._ts)
        mjoint = TMTNJoint(tm,tn)

        # check for phase reference
        latph = utils.get_par_value(self._par,'LATPHASE')
        if latph is not None:
            tm._offset_val = float(latph)
        else:
            # need to align these manually
            print('Brute force aligning profile.')
            profile_align(mjoint,self)

        return mjoint

    def extrapolate_model(self,mjoint,ngrid=512,add_tobs=0):
        """ After, e.g., updating TOAs, update the provided model to span
        the new data range, after *extrapolating* the phase from the model
        to the new dataspan.
        """

        grid_ts,grid_mjds = self.get_grid_ts(ngrid)
        mjoint_extrap = mjoint.change_timerange(grid_ts)
        target_phase = mjoint_extrap.phase()

        tm_tweak = mjoint_extrap.tm()
        nharm = len(mjoint_extrap.tn().freqs)
        tn_tweak = self.get_tn(nharm,grid_mjds=grid_mjds,add_tobs=add_tobs)
        mjoint_tweak = TMTNJoint(tm_tweak,tn_tweak)
        fit_model(mjoint_tweak,target_phase)

        # now move to full photons
        tm_new = tm_tweak.get_fixed(self.get_ts())
        tn_new = tn_tweak.get_fixed(self.get_tdbs())
        mjoint_new = TMTNJoint(tm_new,tn_new)

        return mjoint_new

class PhotonDataSet(DataSet):
    """ Encapsulate a Fermi dataset.
    
    (I suppose this should be called FermiDataSet.  Maybe in the future.)

    Parameters
    ----------
    par : str
        An ephemeris corresponding to an initial timing model.
    photon_ts : pint.toa.TOAs instance
        PINT TOAs corresponding to the Fermi photon timestamps.
    photon_we : float array
        Weights for each photon.
    template : pint.templates.LCTemplate instance
        An analytic template for the pulsar light curve.
    minweight : float
        Keep track of minimum weight used in creation.
    altweight : float array
        Alternative set of weights, e.g. as from a different sky model.
    """

    def __init__(self,par,photon_ts,photon_we,template,minweight=0.05,
            altweight=None):
        super().__init__(par,photon_ts)
        self._we = photon_we
        self._lct = template
        self._minweight = utils.fix_minweight(minweight)
        self._we_alt = altweight
        if altweight is not None:
            if not (len(altweight)==len(photon_we)):
                raise ValueError('Provided alternative weights must have same shape as photon weights')

    def get_minweight(self):
        """ Try to reconstruct initial conditions by looking at the
        minimum weight and returning the nearest version rounded to the
        1e-2 place if it's low and the 1e-3 place if it's high.
        """
        if hasattr(self,'_minweight'):
            return self._minweight
        return utils.fix_minweight(self._we.min())

    def get_weights(self):
        return self._we[self._select_mask]

    def get_altweights(self):
        if self._we_alt is not None:
            return self._we_alt[self._select_mask]

    def get_weight_rate(self):
        return np.sum(self.get_weights())/(self._t1-self._t0)

    def get_template(self,copy=True):
        if copy:
            c = self._lct.copy()
            if c.ncache != self._lct.ncache:
                c.set_cache_properties(ncache=self._lct.ncache)
            return c
        return self._lct

    def estimate_wn_psd(self):
        """ Return a rough estimate of the white noise level based on the
        pulse shape and weights.  This is done by approximating the hessian
        for a sine/cosine term.
        """

        # to make life easy, w_i --> 1 --> H = (fi_prime/fi)^2 * design
        dom = np.linspace(0,1,1001)
        scale = self.get_tn_scale()/2**0.5 # approximate design matrix
        wn_psd = scale*(self._we**2).mean()/np.mean((self._lct.derivative(dom,order=1)/self._lct(dom))**2)
        return wn_psd

    def get_tn_model(self,nharm,max_fn=1,add_tobs=0,
            center_epoch=True,ngrid=2048,deepcopy_TOAs=True,p_tn=None,
            target_model=None,override_par=None,max_recurse=3):
        """ Convert between whatever the internal model is to one suitable
        for doing a timing noise fit.  Assumption is that the internal
        model is sufficiently white (<0.05P or so eveywhere).
        """
        func_kwargs = locals().copy()
        func_kwargs.pop('self')
        func_kwargs.pop('nharm')

        # Evaluate phase difference between internal and clean model on a
        # relatively coarse grid and use fast least squares fit to get the
        # phase difference realized as a TN model (F0/F1 + harmonics)
        # NB this grid is defined at the barycenter
        grid_ts,grid_mjds = self.get_grid_ts(ngrid)
        if override_par is None:
            morig = self._morig
        else:
            morig = utils.get_pint_model(override_par)
        minit,ph_init = clean_model(morig,grid_ts,
                max_fn=max_fn,center_epoch=center_epoch)
        tn_init = self.get_tn(nharm,grid_mjds=grid_mjds,add_tobs=add_tobs,
                waveepoch=minit.PEPOCH.value)
        tm_init = TMParametersFixed(minit,grid_ts)
        mjoint_init = TMTNJoint(tm_init,tn_init)
        if target_model is None:
            target_phase = mjoint_init.phase() + ph_init
        else:
            target_phase = target_model.phase(grid_ts)

        # At this point, we have a model that will match the phase of the 
        # TOAs.  If we want to include simulated timing noise, we must
        # remove this signal from the fit below, then add the signal
        # post facto.
        if self._tn_func is not None:
            target_phase -= self._tn_func(grid_mjds)*self.F0_scale
        fit_ok = fit_initial_model(mjoint_init,target_phase,max_fn=max_fn,
                p_tn=p_tn)
        if not fit_ok:
            if max_recurse > 0:
                print('Could not succeed with nharm=%d.  Trying more.'%nharm)
                new_nharm = int(round(1.5*nharm))
                func_kwargs['max_recurse'] = max_recurse -1
                return self.get_tn_model(new_nharm,**func_kwargs)
            else:
                raise ValueError('Initial model no good!  Failing.')

        # Now produce similar models with the full photon timestamps, and
        # include the simulated timing noise in the signal model if using
        tm_init.set_tn_func(self._tn_func)
        if deepcopy_TOAs:
            tm_fermi = tm_init.get_fixed(grid_ts=deepcopy(self._ts))
        tn_fermi = tn_init.get_fixed(self.get_tdbs())
        mjoint_fermi = TMTNJoint(tm_fermi,tn_fermi)
        return mjoint_fermi

    def template_logl(self,ph,we):
        return np.sum(np.log(we*self._lct(ph%1,use_cache=True)+(1-we)))


class CMatrix(object):
    """ Encapsulate a covariance matrix and convenience methods."""

    def __init__(self,M,inverse=False,diagonal=False,make_float64=False):
        if make_float64:
            M = M.astype(np.float64)
        self.diagonal = diagonal
        if inverse:
            self._H = M
            self._C = None
        else:
            self._C = M
            self._H = None
        self._cho_factor_H = None
        self._cho_factor_C = None

    def logdet(self):
        """ Return sum(log(determinant(C)))."""
        return 2*np.sum(np.log(np.diag(self.cho_fac_C())))

    def cho_fac_H(self):
        if self._cho_factor_H is None:
            if self._H is None:
                self.H()
                return self._cho_factor_H
            #try:
            #print('condition number on H',np.linalg.cond(self._H))
            #self._Hscale = np.diag(self._H)
            #c = cho_factor(self._H/np.outer(self._Hscale,self._Hscale),lower=True)
            c = cho_factor(self._H,lower=True)
            #except LinAlgError:
                #x = np.diag(self._H).min()
                #print('Warning, need to stabilize this matrix.')
                #H = self._H + np.eye(self._H.shape[0])*(x*1e-5)
                #c = cho_factor(H,lower=True)
            self._cho_factor_H = c[0]
        return self._cho_factor_H

    def cho_fac_C(self):
        # todo -- specialize for diagonal case
        if self._cho_factor_C is None:
            if self._C is None:
                self.C()
                return self._cho_factor_C
            c = cho_factor(self._C,lower=True)
            self._cho_factor_C = c[0]
        return self._cho_factor_C

    def eval_MCM(self,M):
        """ Return the congruent matrix M^T C M."""
        q = self.cho_fac_C()@M.T
        return q.T@q

    def C(self):
        """ Return covariance matrix."""
        if self._C is None:
            if self.diagonal:
                self._C = 1./self._H
            else:
                c = self.cho_fac_H()
                L = solve_triangular(c,np.eye(c.shape[0]),
                        overwrite_b=True,lower=True)
                self._cho_factor_C = L
                #self._C = np.matrix(L).T*L
                self._C = L.T@L
        return self._C

    def H(self):
        """ Return inverse covariance matrix."""
        if self._H is None:
            if self.diagonal:
                self._H = 1./self._C
            else:
                c = self.cho_fac_C()
                L = solve_triangular(c,np.eye(c.shape[0]),
                        overwrite_b=True,lower=True)
                self._cho_factor_H = L
                self._H = L.T@L
        return self._H

    def least_squares(self,x,M):
        """ Return coefficients given signal x and design matrix M."""
        # if we already have covariance matrix, use it
        # otherwise, need to cho_factor.  But will have cho_factor anyway...
        b = np.matrix(self.H())*np.matrix(M).T
        A = np.matrix(M)*b
        b = x*b
        c = cho_factor(A,lower=True)
        return cho_solve(c,np.asarray(b)[0])

def HMatrix(H,diagonal=False,make_float64=False):
    return CMatrix(H,inverse=True,diagonal=diagonal,
            make_float64=make_float64)

class LikelihoodFit(object):
    """ Encapsulate a likelihood fit to data."""

    def __init__(self,mjoint,dataset,align=True,G_scale=None,
            H_tn=None):
        self.mjoint = mjoint
        self.process_dataset(dataset)
        if align:
            self.align()
        self.x0 = mjoint.get_values()
        self._last_p = None
        self._last_p2 = None
        self._cache_vals = None
        self._cache_vals2 = None

        self._last_htn = None
        self._marg_cache_vals = None

        self.update_scales(G_scale=G_scale,H_tn=H_tn)
        assert(not np.any(np.isnan(self.H_scale)))

    def update_scales(self,G_scale=None,H_tn=None):
        if G_scale is None:
            ph,fphi,M,G,H = self.update_cache(self.x0)
            # TMP
            debug_locals.update(locals())
            # end TMP
            if H_tn is not None:
                H = H + H_tn
            try:
                C = HMatrix(H).C()
                if np.any(np.diag(C)<0):
                    a = np.argwhere(np.diag(C)<0)
                    print('Negative covariance for parameters:')
                    print(np.asarray(self.mjoint().get_params()[a]))
                    raise ValueError('Found negative covariances in scale.')
                self.G_scale = np.diag(C)**0.5
            except LinAlgError:
                print('Ignoring correlations when estimating parameter scales.')
                if np.any(np.diag(H)<0):
                    debug_locals.update(locals())
                    raise ValueError('Negative entries in H -- should not be possible!  Check alignment of template.')
                self.G_scale = 1./np.diag(H)**0.5
        else:
            self.G_scale = G_scale
        self.H_scale = np.outer(self.G_scale,self.G_scale)

    def process_dataset(self,d):
        raise NotImplementedError

    def data_goodies(self,d):
        raise NotImplementedError

    def align(self):
        raise NotImplementedError

    def update_cache(self,p):
        """ Return phase, template evaluated (if applicable), design
            matrix, gradient, and hessian.  Cache values for additional
            calls.
        """
        if np.all(p==self._last_p):
            return self._cache_vals
        self._last_p = p
        self.mjoint.set_values(p)
        #ph,fphi,M,G,H = self.data_goodies(stabilize=0,use_cache=True)
        ph,fphi,M,G,H = self.data_goodies()
        self._cache_vals = ph,fphi,M,G,H
        return self._cache_vals

    def update_cache2(self,p):
        """ Return phase, template evaluated (if applicable), design
            matrix, gradient, and hessian.  Cache values for additional
            calls.
        """
        if np.all(p==self._last_p2):
            return self._cache_vals2
        self._last_p2 = p
        self.mjoint.set_values(p)
        self._cache_vals2 = self.data_goodies2()
        return self._cache_vals2

    def get_cached_vals(self):
        return self._cache_vals

    def get_p(self,p):
        np = len(p)
        return self.x0[:np] + self.G_scale[:np]*p

    def get_tm_p(self,p):
        ntm_parm = self.mjoint.get_ntm_param()
        return self.x0[:ntm_param] + self.G_scale[:ntm_param]*p

    def physical_grad(self,G):
        """ Return gradient with scale removed."""
        return G/self.G_scale[:len(G)]

    def physical_hess(self,H):
        """ Return hessian with scale removed."""
        n = H.shape[0]
        return H/self.H_scale[:n,:n]

    def update_x0_from_p(self,p):
        p = self.get_p(p)
        self.x0 = p

    def logl(self,p,H_tn=None):
        raise NotImplementedError

    def grad(self,p,H_tn=None):
        p = self.get_p(p)
        ph,fphi,M,G,H = self.update_cache(p)
        if H_tn is not None:
            G = G - np.inner(H_tn,p)
        # negative gradient for negative log likelihood
        if np.any(np.isinf(G)):
            print('oops_inf!',p,G)
        if np.any(np.isnan(G)):
            print('oops_nan!',p,G)
        return -G*self.G_scale

    def grad2(self,p,H_tn=None):
        p = self.get_p(p)
        M,ph,fphi,grad,hess = self.update_cache2(p)
        G = np.inner(M,grad)
        if H_tn is not None:
            G = G - np.inner(H_tn,p)
        return -G*self.G_scale

    def num_grad(self,p,H_tn=None,delta=1e-3):
        rvals = np.empty_like(p)
        for i in range(len(p)):
            pcopy = p.copy()
            pcopy[i] = p[i] + delta
            fhi = self.logl(pcopy,H_tn=H_tn)
            pcopy[i] = p[i] - delta
            flo = self.logl(pcopy,H_tn=H_tn)
            rvals[i] = (fhi-flo)/(2*delta)
        return rvals

    def hess(self,p,H_tn=None):
        """ Return negative hessian.  This is common to gaussian and
        poisson cases because inherited methods take care of generating
        the "data" hessian.
        """
        p = self.get_p(p)
        ph,fphi,M,G,H = self.update_cache(p)
        if H_tn is not None:
            H = H + H_tn
        # NB H is already the negative hessian in this def
        h = H*self.H_scale
        if np.any(np.isnan(h)):
            raise ValueError('nan!')
        return h

    def hess2(self,p,H_tn=None):
        """ Return negative hessian.  This is common to gaussian and
        poisson cases because inherited methods take care of generating
        the "data" hessian.
        """
        p = self.get_p(p)
        M,ph,fphi,grad,hess = self.update_cache2(p)
        H = form_H(M,hess)
        if H_tn is not None:
            H = H + H_tn
        # NB H is already the negative hessian in this def
        h = H*self.H_scale
        return h

    def hessp(self,p,v,H_tn=None):
        #print(p[:10],type(v),type(H_tn))
        p = self.get_p(p)
        v = v*self.G_scale
        M,ph,fphi,grad,hess = self.update_cache2(p)
        Hp = form_Hp(M,hess,v)
        if H_tn is not None:
            Hp += np.inner(H_tn,v)
        return Hp*self.G_scale

    def fit(self,x0=None,H_tn=None,update_position=True,fit_kwargs=dict(),
            method='trust-exact',quiet=True):
        if x0 is None:
            x0 = np.zeros_like(self.x0)
        if 'gtol' not in fit_kwargs.keys():
            fit_kwargs['gtol'] = 1e-2
        if 'maxiter' not in fit_kwargs.keys():
            fit_kwargs['maxiter'] = 200
        initial_logl = self.logl(x0,H_tn=H_tn)
        results = minimize(self.logl,x0,jac=self.grad,hess=self.hess,
                method=method,options=fit_kwargs,args=(H_tn,))          
        # ensure that the state is at the best-fit value
        if not quiet:
            print('Initial logl=%.2f'%initial_logl)
            print('Fit success: %s'%(results['success']),', best-fit log likelihood: %.2f'%(results['fun']),' iterations=%d'%results['nit'])
        results['physical_hess'] = self.physical_hess(results['hess'])
        results['fun0'] = initial_logl
        results['logl_notn'] = self.logl(results['x'],H_tn=None)
        #if update_position and results['success']:
            #self.update_p0_from_p(results['x'])
        return results

    def fit2(self,x0=None,H_tn=None,update_position=True,fit_kwargs=dict(),
            method='trust-krylov',quiet=True):
        if x0 is None:
            x0 = np.zeros_like(self.x0)
        if 'gtol' not in fit_kwargs.keys():
            fit_kwargs['gtol'] = 1e-2
        if 'maxiter' not in fit_kwargs.keys():
            fit_kwargs['maxiter'] = 200
        fit_kwargs['inexact'] = False
        initial_logl = self.logl2(x0,H_tn=H_tn)
        results = minimize(self.logl2,x0,jac=self.grad2,hessp=self.hessp,
                method=method,options=fit_kwargs,args=(H_tn,))          
        hess = self.hess2(results['x'],H_tn=H_tn)
        results['hess'] = hess
        # ensure that the state is at the best-fit value
        logl = self.logl2(results['x'],H_tn=H_tn)
        if not quiet:
            print('Initial logl=%.2f'%initial_logl)
            print('Fit success: %s'%(results['success']),', best-fit log likelihood: %.2f'%(results['fun']),' iterations=%d'%results['nit'])
        results['physical_hess'] = self.physical_hess(results['hess'])
        results['fun0'] = initial_logl
        #if update_position and results['success']:
            #self.update_p0_from_p(results['x'])
        return results

    # TODO -- copy any relevant bits from docstring of old fit_tn
    # (now in old_code.py)
    # TODO -- consider removing production of Htopt -- it is useful?
    def fit_tn(self,p0=None,x0=None,maxiter=200,
            get_covariance=False,minamp=-15):
        """ Fit timing noise amplitude and index.

        The TN amplitude is given as log10(A) and A has units of s^2/yr,
        and value of the PSD at f=1yr.
        
        Parameters
        ----------
        p0 : [float,float]
            Starting TN parameters for fit.
        x0 : float array
            Timing model parameters.  If not provided, will use current
            value of class member x0.
        maxiter : int
            Maximum iterations of fmin routine.
        get_covariance : bool
            Return covariance matrix estimated from finite diff hessian.
        minamp : float
            log10(amp) of PSD.  If best-fit values are below this, treat
            the TN as 0 and return a fixed value, along with 0 covariance.

        Returns
        -------
        popt : [float,float]
            Maximum likelihood TN amplitude (log10) and index.
        Htopt : float matrix
            Inverse covariance matrix associated with the noise process.
        loglopt : float
            Value of log likelihood at best-fit TN model.
        C : 2x2 float matrix [optional]
            Covariance matrix, returned if get_covariance is specified.
        
        
        This is a slightly more general version of fit_tn, where I
        don't assume that a fit with only the white-noise model has been
        done.  This means there's an extra term in the log likelihood
        (from the gradient) which is 0 in the simpler version.  It is
        easier to see this version when starting from the "data" likelihood
        rather than the version expressed in terms of the parameters.
        
        Group terms, the gaussian/linearized likelihood is: 
        -0.5*dlambda*Hdd*dlambda
        -0.5*dalpha*(Hff+Ht)*dalpha
        -0.5*alpha*Ht*alpha
        -1.0*dlambda*(Hdf*dalpha-Gd)
        -1.0*dalpha*(Hfd*dlambda+Ht*alpha-Gf)
        +0.5*log(det(Ht))

        where Gd = resid*Hw*Md and Gf = resid*Hw*Mf, and where I have
        ***repeated the same dlambda*dalpha cross-term for symmetry.***

        Integrate wrt dalpha.  A1 = Hff + Ht, B1 = Gf-Hfd*dlambda-Ht*alpha.
        -0.5*dlambda*(Hdd-Hdf*A1^-1*Hfd)*dlambda
        +1.0*dlambda*(Gd-Hdf*A1^-1*(Gf-Ht*alpha))
        +0.5*(Gf-Ht*alpha)*A1^-1*(Gf-Ht*alpha)
        -0.5*log(det(A1))+0.5*log(det(Ht))-0.5*alpha*Ht*alpha

        Letting y=Gf-Ht*alpha, the marginal likelihood is
        -0.5*dlambda*(Hdd-Hdf*A1^-1*Hfd)*dlambda
        +1.0*dlambda*(Gd-Hdf*A1^-1*y)
        +0.5*y*A1^-1*y -0.5*alpha*Ht*alpha -0.5*r*Hw*r
        -0.5*log(det(A1) +0.5*log(det(Ht))

        Now let A2 = Hdd-Hdf*A1^-1*Hfd, B2 = Gd-Hdf*A1^-1*y
        0.5*(Gd-Hdf*A1^-1*y)A2^-1*(Gd-Hdf*A1^-1*y)
        +0.5*y*A1^-1*y -0.5*alpha*Ht*alpha -0.5*r*Hw*r
        -0.5*log(det(A1))-0.5*log(det(A2)) + 0.5*log(det(Ht))

        I have not yet worked out the version integrating wrt dlambda
        first, yet.  NB that this reduces to the other version if you
        let Gd=Gf=0.

        """

        p = self.x0 if (x0 is None) else self.get_p(x0)
        ph,fphi,M,G,H = self.update_cache(p)
        mjoint = self.mjoint
        Gd,Gf = mjoint.get_subspace_G(G)
        Hdd,Hdf,Hff = mjoint.get_subspace_H(H)
        alpha = mjoint.tn().get_values()

        def logl1(p):
            if p[1] < 0:
                return np.inf
            print(p)
            Ht = mjoint.get_htn(p,zeropad=False)
            # The t1/t2 difference below can be subject to numerical
            # errors if the absolute values become too large (essentially
            # the difference of two very large numbers).  This little
            # check below seems to work pretty well to avoid it.
            t1 = -np.inner(np.diag(Ht),alpha**2)
            if abs(t1) > 1e3*len(alpha):
                print('Check1 fail')
                return np.inf
                #return 1e100
            # end that little check
            try:
                a1 = HMatrix(Hff + Ht)
                c1 = a1.C()
            except LinAlgError:
                print('LinAlgError1')
                return np.inf
                #return 1e100
            try:
                a2 = HMatrix(Hdd - Hdf@c1@Hdf.T)
                c2 = a2.C()
            except LinAlgError:
                print('LinAlgError2')
                return np.inf
                #return 1e100
            y = Gf-np.diag(Ht)*alpha
            z = Gd-y@c1@Hdf.T # == b2
            det_ht = np.sum(np.log(np.diag(Ht)))
            det_a1 = -a1.logdet() # need -ve signs because logdet(C)
            det_a2 = -a2.logdet()
            #t1 = -np.inner(np.diag(Ht),alpha**2)
            t2 = np.inner(np.inner(c1,y),y).item()
            t3 = np.inner(np.inner(c2,z),z).item()
            logl = 0.5*(det_ht - det_a1 - det_a2 + t1 + t2 + t3)
            # put a check on t2 and t3, too
            if abs(t2) > 1e3*len(alpha):
                print('Check2 fail')
                return np.inf
            if abs(t3) > 1e3*len(alpha):
                print('Check3 fail')
                return np.inf
            #if logl > 1e4:
                #print(logl,det_ht,det_a1,det_a2,t1,t2,t3)
            # the only thing this omits is the -0.5*resid*Hw*resid term
            return -logl

        if p0 is not None:
            logl_p0 = logl1(p0)
        else:
            logl_p0 = None
        print('logl_p0 (initial) = ',logl_p0)

        # check for very low initial TN coeffs
        if (p0 is None) and np.all(np.abs(alpha) < 1e-15):
            p0 = [-10,4]

        if p0 is None:
            # do an initial fit with index=4 to make sure the amplitude
            # is in the right ballpark
            wn_level = self.get_wnlevel()
            mask = alpha > wn_level
            if not np.any(mask):
                mask[:2] = True
            Ht = mjoint.get_htn([0,4],zeropad=False)
            resids = np.mean(np.diag(Ht)[mask]*alpha[mask]**2)
            scale = np.log10(2*np.mean(resids))
            p0 = np.asarray([scale,4])

        print('logl_p0 (pre scan) = ',logl1(p0))
        # check and make sure starting value is OK
        #p0_ok = not np.isinf(logl1(p0))
        p0_ok = np.abs(logl1(p0)) < 1e5
        for ind in [4,6,2]:
            if p0_ok:
                break
            for amp in range(-10,4):
                #p0_ok = not np.isinf(logl1([amp,ind]))
                p0_ok = np.abs(logl1([amp,ind])) < 1e5
                if p0_ok:
                    p0 = [amp,ind]
                    break
        if not p0_ok:
            raise ValueError('Found no acceptable starting parameters!')
        print('logl_p0 (post scan) = ',logl1(p0))
        print('p0=',p0)

        popt = fmin(logl1,p0,disp=0,maxfun=maxiter,maxiter=maxiter)

        # check for, essentially, 0 noise level
        if popt[0] <= minamp:
            popt = [minamp,2]
            Ht = mjoint.get_htn(popt,zeropad=True)
            if get_covariance:
                cov = np.asarray([[0.,0.],[0.,0.]])
                return popt,Ht,logl1(popt),cov,logl_p0
            return popt,Ht,logl1(popt),logl_p0

        Ht = mjoint.get_htn(popt,zeropad=True)
        if get_covariance:
            # TODO -- evaluate uncertainty using 2nd derivative.  Just
            # need a scale, I guess.
            damp = 0.05
            dind = 0.05
            p = np.asarray(popt).copy()
            logl_00 = logl1(p)
            p[0] += damp
            logl_h0 = logl1(p)
            p[1] += dind
            logl_hh = logl1(p)
            p = np.asarray(popt).copy()
            p[0] -= damp
            logl_l0 = logl1(p)
            p[1] -= dind
            logl_ll = logl1(p)
            p = np.asarray(popt).copy()
            p[1] += dind
            logl_0h = logl1(p)
            p[0] -= damp
            logl_lh = logl1(p)
            p = np.asarray(popt).copy()
            p[1] -= dind
            logl_0l = logl1(p)
            p[0] += damp
            logl_hl = logl1(p)
            p = np.asarray(popt).copy()
            h = np.empty([2,2])
            h[0,0] = (logl_h0-2*logl_00+logl_l0)/damp**2
            h[1,1] = (logl_0h-2*logl_00+logl_0l)/dind**2
            h[0,1] = h[1,0] = (logl_hh-logl_hl-logl_lh+logl_ll)/(2*damp*2*dind)
            c = np.linalg.inv(h)
            return popt,Ht,logl_00,c,logl_p0
        return popt,Ht,logl1(popt),logl_p0

    def ml_tn(self,p_tn,fit_kwargs=dict()):
        """ Return maximum likelihood params and covariance for given
        timing noise parameters.
        """
        H_tn = self.mjoint.get_htn(p_tn)
        fit_kwargs['H_tn'] = H_tn
        results_tn = self.fit(fit_kwargs=fit_kwargs)
        # best-fit parameters
        p = self.get_p(results_tn['x']).copy()
        # best-fit covariance matrix
        C = HMatrix(results_tn['hess']/self.H_scale).C()
        return results_tn,p,C

    def marg_update_cache(self,p,H_tn):
        """ Return phase, template evaluated (if applicable), design
            matrix, gradient, and hessian.  Cache values for additional
            calls.
        """
        if np.all(p==self._last_p) and np.all(np.diag(H_tn)==self._last_htn):
            return self._marg_cache_vals
        self._last_p = p
        self._last_htn = np.diag(H_tn)
        self.mjoint.set_values(p)
        ph,fphi,M,G,H = self.data_goodies()

        Hdd,Hdf,Hff = self.mjoint.get_subspace_H(H)

        a1 = HMatrix(Hff + H_tn)

        self._marg_cache_vals = ph,fphi,M,G,H,a1
        return self._marg_cache_vals

    def get_marg_p(self,p):
        ntm_param = len(p)
        return self.get_p(np.append(p,np.zeros(len(self.x0)-ntm_param)))

    def grad_marg_tn(self,p,H_tn):
        """ Return the gradient of the marginalized likelihood.

        Letting y=Gf-Ht*alpha, the marginal likelihood is
        -0.5*dlambda*(Hdd-Hdf*A1^-1*Hfd)*dlambda
        +1.0*dlambda*(Gd-Hdf*A1^-1*y)
        +0.5*y*A1^-1*y -0.5*alpha*Ht*alpha -0.5*r*Hw*r
        -0.5*log(det(A1) +0.5*log(det(Ht))

        We can get the gradient from above by keeping the dlambda
        differential terms, viz
        grad = 1.0*(Gd-Hdf*A1^-1*y)

        """
        ntm_param = len(p)
        p = self.get_p(np.append(p,np.zeros(len(self.x0)-ntm_param)))
        ph,fphi,M,G,H,a1 = self.marg_update_cache(p,H_tn)
        Gd,Gf = self.mjoint.get_subspace_G(G)
        Hdd,Hdf,Hff = self.mjoint.get_subspace_H(H)
        alpha = self.mjoint.tn().get_values()

        #a1 = HMatrix(Hff + H_tn)
        c1 = np.asarray(a1.C())
        y = Gf-np.diag(H_tn)*alpha
        grad = Gd-np.dot(Hdf.dot(c1),y)
        return -grad*self.G_scale[:ntm_param]

    def hess_marg_tn(self,p,H_tn):
        """ Return the hessian of the marginalized matrix.

        Letting y=Gf-Ht*alpha, the marginal likelihood is
        -0.5*dlambda*(Hdd-Hdf*A1^-1*Hfd)*dlambda
        +1.0*dlambda*(Gd-Hdf*A1^-1*y)
        +0.5*y*A1^-1*y -0.5*alpha*Ht*alpha -0.5*r*Hw*r
        -0.5*log(det(A1) +0.5*log(det(Ht))

        We can get the hessian from above by keeping the dlambda^2
        differential terms, viz
        hess = -1.0*(Hdd-Hdf*A1^-1*Hfd)

        """
        ntm_param = len(p)
        p = self.get_p(np.append(p,np.zeros(len(self.x0)-ntm_param)))
        ph,fphi,M,G,H,a1 = self.marg_update_cache(p,H_tn)
        Hdd,Hdf,Hff = self.mjoint.get_subspace_H(H)

        #Ht = H_tn
        #a1 = HMatrix(Hff + Ht)
        c1 = np.asarray(a1.C())
        hess = Hdd-np.inner(np.inner(Hdf,c1),Hdf)
        return hess*self.H_scale[:ntm_param,:ntm_param]
         
    def fit_marg_tn(self,p_tn,x0=None,fit_kwargs=dict(),quiet=True):
        H_tn = self.mjoint.get_htn(p_tn,zeropad=False)
        if x0 is None:
            x0 = np.zeros(self.mjoint.get_ntm_param())
        if 'gtol' not in fit_kwargs.keys():
            fit_kwargs['gtol'] = 1e-2
        results = minimize(self.logl_marg_tn,x0,jac=self.grad_marg_tn,
                hess=self.hess_marg_tn,args=(H_tn),
                method='trust-exact',options=fit_kwargs)
        if not quiet:
            print('Fit success: %s'%(results['success']),', best-fit marginal log likelihood: %.2f'%(results['fun']))
        results['physical_hess'] = self.physical_hess(results['hess'])
        #if update_position and results['success']:
            #self.update_p0_from_p(results['x'])
        return results

    def get_wnlevel(self,x0=None):
        if x0 is None:
            x0 = np.zeros_like(self.x0)
        nharm = self.mjoint.get_nharm()
        nparam_to_use = max(1,nharm//2)
        t = np.diag(2./self.physical_hess(self.hess(x0)))
        return np.median(t[-nparam_to_use:])

    def get_covariance(self,x0=None,physical_hess=None,require_full=True,
            ignore_tn=False):
        """ Using either parameter values, physical hessian, or current
        parameters, calculate the uncertainties for the parameters from the
        covariance matrix.

        Parameters
        ----------
        require_full : bool
            If unable to invert full covariance matrix, return inverse
            diagonal elements if False, else raise exception.
        ignore_tn : bool
            Ignore entries corresponding to timing noise, e.g. in cases
            where it is disabled and the parameters are likely to be
            unconstrained / not at local maxima.
        """
        if physical_hess is not None:
            HM = HMatrix(physical_hess)
        else:
            if x0 is None:
                x0 = np.zeros_like(self.x0)
            HM = HMatrix(self.physical_hess(self.hess(x0)))
        if ignore_tn:
            idx_mask = ~np.asarray([('COS' in x) or ('SIN' in x) for x in self.mjoint.get_params()])
            HM_orig = HM
            HM = HMatrix(HM_orig.H()[idx_mask,:][:,idx_mask])
        try:
            C = HM.C()
        except LinAlgError as e:
            if require_full:
                raise e
            else:
                print('WARNING! Could not estimate covariances.  Using independent parameter approximation.')
                C = np.diag(1./np.diag(HM.H()))
        if ignore_tn:
            C_orig = np.zeros_like(HM_orig.H())
            C_orig[np.outer(idx_mask,idx_mask)] = np.ravel(C)
            C_orig[~np.outer(idx_mask,idx_mask)] = np.inf # or 0?
            C = C_orig
        return C

    def set_pint_uncertainties(self,x0=None,physical_hess=None,
            ignore_tn=False):
        """ Update the uncertainties in the underlying model using either
        a best-fit result, a physical hessian, or the current position.

        Parameters
        ----------
        ignore_tn : bool
            Ignore entries corresponding to timing noise, e.g. in cases
            where it is disabled and the parameters are likely to be
            unconstrained / not at local maxima.
        """
        C = self.get_covariance(x0=x0,physical_hess=physical_hess,
                require_full=False,ignore_tn=ignore_tn)
        tm_params = self.mjoint.tm().get_params()
        uncertainties = (np.diag(C)[:len(tm_params)])**0.5
        model = self.mjoint.tm().model
        for param,err in zip(tm_params,uncertainties):
            if param == 'Offset':
                continue
            # some parameters have different units for uncertainty and
            # value, e.g. RAJ/DECJ.  Other parameters don't even have
            # units (TASC, e.g., 
            # need to match units -- e.g. for RAJ/DECJ
            p = getattr(model,param)
            if not hasattr(p,'uncertainty'):
                print('No uncertainty quantity for parameter %s!'%param)
                continue
            # default behavior is to assume units are the same
            p.uncertainty = err * p.uncertainty.unit
            # quantity is present, use its unit instead (probably the same)
            if hasattr(p.quantity,'unit'):
                e = (err * p.quantity.unit).to(p.uncertainty.unit)
                p.uncertainty = e

class PhotonLikelihoodFit(LikelihoodFit):
    """ Encapsulate a Fermi likelihood fit."""

    def process_dataset(self,dataset):
        self.lct = dataset.get_template()
        assert(self.lct.ncache > 10000)
        self.weights = dataset.get_weights()

    def align(self):
        """ Set offset value to match a template."""
        ph = self.mjoint.modphase()
        # KLUGE -- ideally cache should survive a copy!  need to fix in
        # PINT
        lct = self.lct.copy()
        lct.set_cache_properties(ncache=20000)
        # end kluge
        lcf = lcfitters.LCFitter(lct,ph,weights=self.weights)
        dphi,dphie = lcf.fit_position(unbinned=False)
        self.mjoint.tm().update_offset(dphi)

    def data_goodies(self,stabilize=0,use_cache=True,float64=True):
        M = self.mjoint.designmatrix(float64=float64)
        ph = self.mjoint.modphase()
        fphi = self.lct(ph,use_cache=use_cache)
        gphi = self.lct.derivative(ph,use_cache=use_cache,order=1)
        hphi = self.lct.derivative(ph,use_cache=use_cache,order=2)
        t = self.weights/(self.weights*fphi+(1-self.weights))
        grad = gphi*t
        hess = grad**2-hphi*t
        #G = np.einsum('ij,j',M,grad)
        G = np.inner(M,grad)
        # TODO -- test this formulation
        # H = np.asarray((np.asarray(M)*(grad**2-hphi*t))*M.T)
        #H = np.einsum('ji,ki,i->jk',M,M,grad**2-hphi*t,optimize=True)
        H = form_H(M,hess)
        if stabilize > 0:
            H += np.diag(np.diag(H)*stabilize)
        return ph,fphi,M,G,H

    def data_goodies2(self,use_cache=True,float64=True):
        M = self.mjoint.designmatrix(float64=float64)
        ph = self.mjoint.modphase()
        fphi = self.lct(ph,use_cache=use_cache)
        gphi = self.lct.derivative(ph,use_cache=use_cache,order=1)
        hphi = self.lct.derivative(ph,use_cache=use_cache,order=2)
        t = self.weights/(self.weights*fphi+(1-self.weights))
        grad = gphi*t
        hess = grad**2-hphi*t
        return M,ph,fphi,grad,hess

    def logl(self,p,H_tn=None):
        p = self.get_p(p)
        ph,fphi,M,G,H = self.update_cache(p)
        we = self.weights
        logl = np.sum(np.log(we*fphi+(1-we)))
        if H_tn is not None:
            logl -= 0.5*np.sum(np.diag(H_tn)*p**2)
        # negative log likelihood
        return -logl

    def logl2(self,p,H_tn=None):
        p = self.get_p(p)
        M,ph,fphi,grad,hess = self.update_cache2(p)
        we = self.weights
        logl = np.sum(np.log(we*fphi+(1-we)))
        if H_tn is not None:
            logl -= 0.5*np.sum(np.diag(H_tn)*p**2)
        # negative log likelihood
        return -logl

    def logl_marg_tn(self,p,H_tn):
        """ Return the likelihood marginalized over the timing-noise
        parameters.

        See notes in GaussianLikelihoodFit on derivation -- it's basically
        the same, except "t3" (which is the quadratic form of the
        measurement noise matrix with the residuals) becomes the 
        Poisson likelihood evaluated at the current parameters.
        """
        print('Warning -- this function is not accurate if called with parameters far from the maximum likelihood value, because the logl/grad/hess will not be calculated correctly because the phase is too bad to keep the profile going.')
        p = self.get_marg_p(p)
        ph,fphi,M,G,H,a1 = self.marg_update_cache(p,H_tn)
        Gd,Gf = self.mjoint.get_subspace_G(G)
        Hdd,Hdf,Hff = self.mjoint.get_subspace_H(H)
        alpha = self.mjoint.tn().get_values()
        logl0 = np.sum(np.log(self.weights*fphi+(1-self.weights)))
        t2 = -np.inner(np.diag(H_tn),alpha**2)
        if abs(t2) > 1e2*len(alpha):
            #return 1e100
            return np.inf

        # just keep the defintion of A1 here for convenience
        #a1 = HMatrix(Hff + Ht)
        c1 = a1.C()
        y = Gf-np.diag(H_tn)*alpha
        det_ht = np.sum(np.log(np.diag(H_tn)))
        det_a1 = -a1.logdet() # need -ve signs because logdet(C)
        t1 = np.inner(np.inner(c1,y),y).item()
        #t3 = -np.inner(self.h_vec,fphi**2)
        logl = 0.5*(det_ht - det_a1 + t1 + t2)
        return -logl -logl0

class BinnedPhotonLikelihoodFit(PhotonLikelihoodFit):

    def __init__(self,mjoint,dataset,tbin_or_edges,phase_bins=1000,
            align=True,G_scale=None,H_tn=None):
        """ tbin_or_edges: if scalar, a tbin in days.  Otherwise, the
            edges for the binning.
            """
        if np.ndim(tbin_or_edges)==0:
            # interpret as tbin and generate edges
            nbin = int(np.ceil((dataset._t1-dataset._t0)/tbin_or_edges))
            self.edges = np.linspace(dataset._t0,dataset._t1,nbin+1)
        else:
            self.edges = tbin_or_edges
        self._phase_bins = phase_bins
        super().__init__(mjoint,dataset,
                align=align,G_scale=G_scale,H_tn=H_tn)

    def process_dataset(self,dataset):
        # divide data up into bins (assigned by topocentric time) and make
        # a likelihood profile for each bin,
        d = dataset
        old_t0,old_t1 = d._t0,d._t1
        d.set_timerange(self.edges[0],self.edges[-1])
        ph = self.mjoint.modphase(d.get_ts())
        we = d.get_weights()
        mjds = d.get_mjds()
        d.set_timerange(d._t0,d._t1)
        nbin = len(self.edges)-1
        idx = np.searchsorted(self.edges,mjds)-1
        assert(idx[0] >= 0)
        assert(idx[-1] < nbin)
        cts = np.bincount(idx,minlength=nbin)
        assert(cts.sum()==len(ph))

        phoffs = np.linspace(0,1,self._phase_bins+1)[:-1]
        logls = np.empty((nbin,self._phase_bins))
        logld1s = np.empty_like(logls)
        logld2s = np.empty_like(logls)

        print('We begin')
        lct = d.get_template()
        i0 = 0
        for i in range(nbin):
            if cts[i] == 0:
                logls[i] = 0
                continue
            p = ph[i0:i0+cts[i]]
            w = we[i0:i0+cts[i]]
            wi = 1-w
            for j in range(self._phase_bins):
                this_ph = (p+phoffs[j])%1
                f0 = lct(this_ph,use_cache=True)
                f1 = lct.derivative(this_ph,order=1,use_cache=True)
                f2 = lct.derivative(this_ph,order=2,use_cache=True)
                t = w*f0 + wi
                x = w/t
                logls[i,j] = np.sum(np.log(t))
                logld1s[i,j] = np.sum(x*f1)
                logld2s[i,j] = np.sum(f2*x-(f1*x)**2)

            i0 += cts[i]
        print('We end')

        self._logls = logls
        self._logld1s = logld1s
        # do this to follow sign convention of other implementations
        self._logld2s = -logld2s

        # TODO -- add spline interpolators or just do lookup?
        x = np.linspace(0,1,self._phase_bins+1)
        self._cs0 = [CubicSpline(x,np.append(y,y[0]),bc_type='periodic') for y in self._logls]
        self._cs1 = [CubicSpline(x,np.append(y,y[0]),bc_type='periodic') for y in self._logld1s]
        self._cs2 = [CubicSpline(x,np.append(y,y[0]),bc_type='periodic') for y in self._logld2s]

        # manage the phase
        grid_mjds = 0.5*(self.edges[:-1]+self.edges[1:])
        grid_ts = pint_toas_from_mjds(grid_mjds,
                self.mjoint.pm().EPHEM.value,site='geo')[0]
        grid_ph = self.mjoint.modphase(grid_ts)
        self._grid_mjds = grid_mjds
        self._grid_ts = grid_ts
        self._grid_ph = grid_ph
        self.mjoint_binned = self.mjoint.get_fixed(self._grid_ts)

        super().process_dataset(dataset)

    def data_goodies(self,stabilize=0,use_cache=True,float64=True):
        M = self.mjoint_binned.designmatrix(float64=float64)
        ph = self.mjoint_binned.modphase()
        dphi = (ph-self._grid_ph)%1
        logl = np.asarray([cs(x) for cs,x in zip(self._cs0,dphi)])
        grad = np.asarray([cs(x) for cs,x in zip(self._cs1,dphi)])
        hess = np.asarray([cs(x) for cs,x in zip(self._cs2,dphi)])
        G = np.inner(M,grad)
        H = form_H(M,hess)
        if stabilize > 0:
            H += np.diag(np.diag(H)*stabilize)
        return dphi,logl,M,G,H

    def data_goodies2(self,use_cache=True,float64=True):
        M = self.mjoint_binned.designmatrix(float64=float64)
        ph = self.mjoint_binned.modphase()
        dphi = (ph-self._grid_ph)%1
        print(dphi)
        logl = np.asarray([cs(x) for cs,x in zip(self._cs0,dphi)])
        grad = np.asarray([cs(x) for cs,x in zip(self._cs1,dphi)])
        hess = np.asarray([cs(x) for cs,x in zip(self._cs2,dphi)])
        return M,dphi,logl,grad,hess


    def logl(self,p,H_tn=None):
        p = self.get_p(p)
        dphi,logl,M,G,H = self.update_cache(p)
        logl = np.sum(logl)
        if H_tn is not None:
            logl -= 0.5*np.sum(np.diag(H_tn)*p**2)
        # negative log likelihood
        return -np.sum(logl)

    def logl2(self,p,H_tn=None):
        p = self.get_p(p)
        M,dphi,logl,grad,hess = self.update_cache2(p)
        logl = np.sum(logl)
        if H_tn is not None:
            logl -= 0.5*np.sum(np.diag(H_tn)*p**2)
        # negative log likelihood
        return -logl

    def align(self):
        raise NotImplementedError

    def logl_marg_tn(self,p,H_tn):
        raise NotImplementedError
        """ Return the likelihood marginalized over the timing-noise
        parameters.

        See notes in GaussianLikelihoodFit on derivation -- it's basically
        the same, except "t3" (which is the quadratic form of the
        measurement noise matrix with the residuals) becomes the 
        Poisson likelihood evaluated at the current parameters.
        """
        print('Warning -- this function is not accurate if called with parameters far from the maximum likelihood value, because the logl/grad/hess will not be calculated correctly because the phase is too bad to keep the profile going.')
        p = self.get_marg_p(p)
        ph,fphi,M,G,H,a1 = self.marg_update_cache(p,H_tn)
        Gd,Gf = self.mjoint.get_subspace_G(G)
        Hdd,Hdf,Hff = self.mjoint.get_subspace_H(H)
        alpha = self.mjoint.tn().get_values()
        logl0 = np.sum(np.log(self.weights*fphi+(1-self.weights)))
        t2 = -np.inner(np.diag(H_tn),alpha**2)
        if abs(t2) > 1e2*len(alpha):
            #return 1e100
            return np.inf

        # just keep the defintion of A1 here for convenience
        #a1 = HMatrix(Hff + Ht)
        c1 = a1.C()
        y = Gf-np.diag(H_tn)*alpha
        det_ht = np.sum(np.log(np.diag(H_tn)))
        det_a1 = -a1.logdet() # need -ve signs because logdet(C)
        t1 = np.inner(np.inner(c1,y),y).item()
        #t3 = -np.inner(self.h_vec,fphi**2)
        logl = 0.5*(det_ht - det_a1 + t1 + t2)
        return -logl -logl0

class GaussianLikelihoodFit(LikelihoodFit):
    """ Encapsulate a TOA likelihood fit."""

    def process_dataset(self,dataset):
        # get white noise measurements and pulse numbers
        self._pn = dataset.get_pulse_numbers()
        errs = dataset.get_measurement_errors()
        self.h_vec = 1./errs**2
        self.Hw = np.diag(self.h_vec)

    def align(self):
        pass

    def data_goodies(self):
        ph = self.mjoint.phase()
        # let fph stand in for residual here, why not?
        fph = self._pn - ph
        M = self.mjoint.designmatrix()
        # G is design matrix * error matrix * residuals
        G = np.inner(M,fph*self.h_vec)
        # this is "Hlambda" elsewhere
        H = np.asarray((M*self.h_vec)@M.T)
        return ph,fph,M,G,H

    def logl(self,p,H_tn=None):
        p = self.get_p(p)
        ph,fphi,M,G,H = self.update_cache(p)
        # NB -- fphi == resid = pn-model
        logl = -0.5*np.inner(fphi**2,self.h_vec)
        if H_tn is not None:
            logl -= 0.5*np.inner(np.diag(H_tn),p**2)
        # negative log likelihood
        return -logl

    def logl_marg_tn(self,p,H_tn):
        """ Return the likelihood marginalized over the timing-noise
        parameters.

        NB this wouldn't necessarily be consistent, since the TN model is
        linearized about a particular point -- EXCEPT, the TN model is
        actually linear, so even though a particular value of the TN
        coeffs ("alpha") is used below, it doesn't really matter.

        Letting y=Gf-Ht*alpha, the marginal likelihood is
        -0.5*dlambda*(Hdd-Hdf*A1^-1*Hfd)*dlambda
        +1.0*dlambda*(Gd-Hdf*A1^-1*y)
        +0.5*y*A1^-1*y -0.5*alpha*Ht*alpha -0.5*r*Hw*r
        -0.5*log(det(A1) +0.5*log(det(Ht))

        We can get the likelihood from above by simply letting the
        differential dlambda terms --> 0. Viz
        logL = +0.5*y*A1^-1*y -0.5*alpha*Ht*alpha -0.5*r*Hw*r
               -0.5*log(det(A1) +0.5*log(det(Ht))
        """
        p = self.get_marg_p(p)
        ph,fphi,M,G,H,a1 = self.marg_update_cache(p,H_tn)
        Gd,Gf = self.mjoint.get_subspace_G(G)
        Hdd,Hdf,Hff = self.mjoint.get_subspace_H(H)
        alpha = self.mjoint.tn().get_values()
        t2 = -np.inner(np.diag(H_tn),alpha**2)
        # we have another case where t1 and t2 (here swapped) are in
        # tension and if they get too large their difference becomes much
        # larger than 0, so do a check on the magnitude of the term
        if abs(t2) > 1e2*len(alpha):
            #return 1e100
            return np.inf

        # just keep the defintion of A1 here for convenience
        #a1 = HMatrix(Hff + Ht)
        c1 = a1.C()
        y = Gf-np.diag(H_tn)*alpha
        det_ht = np.sum(np.log(np.diag(H_tn)))
        det_a1 = -a1.logdet() # need -ve signs because logdet(C)
        t1 = np.inner(np.inner(c1,y),y).item()
        #t2 = -np.inner(np.diag(H_tn),alpha**2)
        t3 = -np.inner(self.h_vec,fphi**2)
        logl = 0.5*(det_ht - det_a1 + t1 + t2 + t3)
        return -logl


def time_domain_covariance_matrix(tdbs,p0,flo_fac=128**-0.5,fhi=365):
    """ Times in days (TDBs).
        tn_vals = 2./tn_utils.eval_pl(p,self._freqs)
    """
    # this sets minimum frequency and thus total power.  Needs to be
    # consistent with simulated data if results are to make sense!
    # TODO -- really need to put an explicit cutoff frequency in the
    # simulation.
    flo = 365.25/(tdbs.max()-tdbs.min())*flo_fac # NB in cycles/year
    integration_domain = np.logspace(np.log10(flo),np.log10(fhi),501)
    sf = tn_utils.eval_pl(p0,integration_domain)
    rvals = np.empty((len(tdbs),len(tdbs)))
    for i in range(len(tdbs)):
        #lags = np.abs(tdbs[i] - tdbs[i:])*(np.pi/365.25)
        lags = np.abs(tdbs[i] - tdbs[i:])*(2*np.pi/365.25)
        #lags = np.abs(tdbs[i] - tdbs[i:])*(1./365.25)
        # use FT to compute the time domain values
        for j,lag in enumerate(lags):
            rvals[i,j+i] = simps(sf*np.cos(integration_domain*lag),
                    x=integration_domain)
            rvals[j+i,i] = rvals[i,j+i]
    return rvals

def time_domain_covariance_matrix_uniform(dt,
        n,p0,flo_fac=128**-0.5,fhi=365,nsimps=500):
    """ dt = sample spacing (days), n = num samples
        tn_vals = 2./tn_utils.eval_pl(p,self._freqs)
    """
    # this sets minimum frequency and thus total power.  Needs to be
    # consistent with simulated data if results are to make sense!
    # TODO -- really need to put an explicit cutoff frequency in the
    # simulation.
    flo = 365.25/(dt*n)*flo_fac
    integration_domain = np.logspace(np.log10(flo),np.log10(fhi),nsimps+1)
    sf = tn_utils.eval_pl(p0,integration_domain)
    rvals = np.empty((n,n))
    lags = np.arange(n)*(2*np.pi*dt/365.25)
    ivals = [simps(sf*np.cos(integration_domain*lag),x=integration_domain) for lag in lags]
    for i in range(n):
        for j in range(i,n):
            rvals[i,j] = rvals[j,i] = ivals[j-i]
    return rvals

def form_H(M,q):
    """
        M = design matrix, shape nparam x ndata.
        q = profile information, shape ndata
    """
    output = np.empty((M.shape[0],M.shape[0]),dtype=M.dtype)
    x = M*q
    for i in range(M.shape[0]):
        for j in range(i,M.shape[0]):
            output[i,j] = np.inner(x[i],M[j])
            output[j,i] = output[i,j]
    return output

def form_Hp(M,q,v):
    x = M*q
    t = np.inner(x.transpose(),v)
    return np.inner(M,t)

def load_merged_toas(jname):
    """ Load pre-prepared weekly TOAs."""
    toadir = basedir + '/%s/toas'%(jname)
    t1 = time.time()
    toa_fnames = sorted(glob.glob(toadir + '/*.pickle'))
    if len(toa_fnames) == 0:
        raise ValueError('Found no TOAs for %s!'%jname)
    toas = [pickle.load(open(x,'rb')) for x in toa_fnames]
    toas = toa.merge_TOAs(toas)
    t2 = time.time()
    print('Required %.2f seconds to load and merge %d TOAs.'%(t2-t1,toas.ntoas))
    return toas

def get_dataset(jname,clobber=False,minweight=None,use_tnfit_par=False,
        override_par=None,use_tnfit_lct=True,override_lct=None,quiet=False):
    """ Build/load a dataset for pulsar PSRjname.  Pickle results.

    Parameters
    ----------
    jname : str
        Name of pulsar, must match local directory structure
    clobber : bool
        Delete any existing pickled version.
    minweight : float [0.05]
        Optional float specifying the minimum photon weight to use.
        If provided, it is compared to any pre-existing object and that
        object is clobbered if it doesn't match.
        If the option is not provided and there is no pre-existing object,
        a default cut of 0.05 is adopted.

    Returns
    -------
    d : PhotonDataSet
    """

    timingdir = basedir + '/%s/timing'%(jname)
    par = timingdir + '/%s.par'%(jname)
    if use_tnfit_par:
        tnfit_par = timingdir + '/%s_tnfit.par'%(jname)
        if os.path.isfile(tnfit_par):
            par = tnfit_par
    if override_par is not None:
        par = override_par
    if not quiet:
        print('Using parfile %s.'%par)

    lct_file = timingdir + '/%s_template.pickle'%(jname)
    if use_tnfit_lct:
        tnfit_lct_file = timingdir + '/%s_tnfit_template.pickle'%(jname)
        if os.path.isfile(tnfit_lct_file):
            lct_file = tnfit_lct_file
    if override_lct is not None:
        lct_file = override_lct
    if not quiet:
        print('Using %s for template.'%lct_file)
    lct = pickle.load(open(lct_file,'rb'))

    # TODO -- I think this will only pick it up on the first instance
    # where we have created the object, so get the template minweight
    # successfully.  Subsequent uses will get the ephemeris versions.
    if (minweight is None) and hasattr(lct,'minweight'):
        minweight = lct.minweight

    # really really, really fix this up...
    if (minweight is None):
        minweight = get_par_minweight(par)

    if not clobber:
        try:
            fname = timingdir + '/%s_pipeline.pickle'%jname
            p = pickle.load(open(
                timingdir + '/%s_pipeline.pickle'%jname,'rb'))
            if not quiet:
                print('loaded pickle')
            if not p._par == par:
                #print('not clobber set_par')
                p.set_par(par,quiet=quiet)
            # replace the template, potentially
            p._lct = lct
            # compare the inferred minimum weight with both the specified
            # and the recorded version in the .par file if either present
            pmw = p.get_minweight()
            cmp_mw = minweight or get_par_minweight(p._par)
            #print(pmw)
            #print(cmp_mw)
            if (cmp_mw is not None) and (abs(cmp_mw-pmw)>1e-5):
                print('Minweight %.6f does not match specified value %.6f.'%(pmw,cmp_mw))
                raise IOError
            # fix clock corrections if needed
            ts = p.get_ts()
            ts.clock_corr_info['include_bipm'] = False
            ts.clock_corr_info['include_gps'] = False
            # TODO
            # add in a little kluge here to adjust any existing START times to
            # before the beginning of LAT data, if they are close
            if (p._t0 > 54682) and (abs(p._t0-54682)<7):
                p.set_timerange(54682,p._t1)
            return p
        except (IOError,EOFError):
            print('could not load pickle')
            pass

    # TODO -- need to have consistent minimum weight with data section!

    photon_ts = load_merged_toas(jname)
    we = np.asarray(photon_ts.table['weight']).copy()

    # TODO -- really, really, really need to fix this to be consistent
    if minweight is None:
        minweight = 0.05

    if minweight is not None:
        mask = we >= minweight
        photon_ts.select(mask)
        # will this work??
        photon_ts.table_selects = []
        we = we[mask]
    else:
        minweight = 0.05

    d = PhotonDataSet(par,photon_ts,we,lct,minweight=minweight,
            altweight=None)

    # TODO
    # add in a little kluge here to adjust any existing START times to
    # before the beginning of LAT data, if they are close
    if (d._t0 > 54682) and (abs(d._t0-54682)<7):
        d.set_timerange(54682,d._t1)

    pickle.dump(d,open(timingdir + '/%s_pipeline.pickle'%jname,'wb'),
        protocol=pickle.HIGHEST_PROTOCOL)

    return d

def get_tnfit_results(jname):
    par = timingdir + '/%s_tnfit.par'%(jname)
    if not os.path.isfile(par):
        raise ValueError('Cannot find output of TNFIT procedure.')
    d = common.get_dataset(jname,use_tnfit_par=True,use_tnfit_lct=True)
    model = d.get_tn_model_exact()
    return d,model

def load_data_and_model(jname,use_tnfit=True,override_par=None,
        require_tnfit=False,minweight=None,clobber=False,
        ignore_bounds=False,override_lct=None,override_nharm=None,
        quiet=False):

    t1 = time.time()
    d = get_dataset(jname,use_tnfit_par=True,use_tnfit_lct=True,
            override_par=override_par,minweight=minweight,
            override_lct=override_lct,clobber=clobber,quiet=quiet)
    # TMP? try finer cache settings
    d.get_template(copy=False).set_cache_properties(ncache=20000)
    if ignore_bounds:
        d.clear_timerange()
    if is_tnfit_par(d._par):
        if not quiet:
            print('Loading TNFIT style model.')
        model = d.get_tn_model_exact()
    else:
        if require_tnfit:
            raise ValueError('Ephemeris is not a valid TNFIT output!')
        default_nharm = 10 if (d._morig.F0.value > 100) else 30
        if override_nharm is not None:
            default_nharm = override_nharm
        nharm = get_par_nharm(d._par,default=default_nharm)
        max_fn = get_par_max_fn(d._par)
        #popt,popte = get_par_tnparams(d._par)
        model = d.get_tn_model(
                nharm,center_epoch=False,max_fn=max_fn,p_tn=None)
        profile_align(model,d)


    popt,popte = get_par_tnparams(d._par)
    t2 = time.time()
    if not quiet:
        print('Required %.2f seconds to load_data_and_model with %d TOAs.'%(t2-t1,d.get_ts().ntoas))
    return d,model,(popt,popte)

def get_par_tnparams(par):
    """ Search either the provided or resident ephemeris for timing
        noise parameters and return them if present.
    """
    lat_idx = None
    lat_amp = None
    lat_idxe = None
    lat_ampe = None
    for line in open(par).readlines():
        if 'LATNOTN' in line:
            toks = line.strip().split()
            if int(toks[1]) != 0:
                return [-100,0],[0,0]
    for line in open(par).readlines():
        if 'LATIDX' in line:
            toks = line.strip().split()
            lat_idx = float(toks[1])
            if len(toks) == 3:
                lat_idxe = float(toks[2])
            continue
        if 'LATAMP' in line:
            toks = line.strip().split()
            lat_amp = float(toks[1])
            if len(toks) == 3:
                lat_ampe = float(toks[2])
            continue
    if (lat_idx is not None) and (lat_amp) is not None:
        if lat_ampe is not None:
            return [lat_amp,lat_idx],[lat_ampe,lat_idxe]
        return [lat_amp,lat_idx],None
    return None,None

def get_par_nharm(par,default=40):
    for line in open(par).readlines():
        if 'LATNHARM' in line:
            lat_nharm = int(line.strip().split()[1])
            return lat_nharm
    return default

def get_par_minweight(par):
    for line in open(par,'r').readlines():
        if 'LATMINWT' in line:
            minwt = utils.fix_minweight(float(line.strip().split()[1]))
            return minwt
    return None

def is_tnfit_par(par):
    """ Determine whether an ephemeris has results of a TN fit.
    """
    if utils.get_par_value(par,'LATAMP') is None:
        return False
    if utils.get_par_value(par,'LATIDX') is None:
        return False
    if utils.get_par_value(par,'LATNHARM') is None:
        return False
    # must have a specification of maximum Taylor, LATF2 is legacy
    c1 = utils.get_par_value(par,'LATF2') is None
    c2 = utils.get_par_value(par,'LATFN') is None
    if c1 and c2:
        return False
    # check that number of spindown terms matches specification
    max_fn = get_par_max_fn(par)
    pres_fn = 0
    for i in range(12):
        attr = 'F%d'%(i)
        if utils.get_par_value(par,attr) is None:
            break
        else:
            pres_fn = i
    if max_fn != pres_fn:
        return False
    return True

def get_par_allow_f2(par):
    for line in open(par).readlines():
        if 'LATF2' in line:
            return bool(int(line.strip().split()[1]))
    return False

def get_par_max_fn(par):
    for line in open(par).readlines():
        # legacy
        if 'LATF2' in line:
            return 2
        elif 'LATFN' in line:
            return int(line.split()[1])
    return 1

def profile_align(model,dataset):
    """ Set offset value to match a template."""
    ph = model.modphase()
    lcf = lcfitters.LCFitter(dataset.get_template(),ph,
            weights=dataset.get_weights())
    dphi,dphie = lcf.fit_position(unbinned=False)
    model.tm().update_offset(dphi)

# TODO -- need to propagate LATNOTN?
def as_tnfit_parfile(model,dataset,p_tn,p_tne=None,logl=None,htest=None,
        wn_level=None):
    max_fn = model.get_max_fn()
    additional_lines = [
        'LATMINWT %.6f'%dataset.get_minweight(),
        'LATPHOT %d'%(len(dataset.get_weights())),
        #'LATF2 %d'%(1 if allow_f2 else 0)]
        'LATFN %d'%(max_fn)]
    if logl is not None:
        additional_lines.append('LATLOGL %.3f'%logl)
    if htest is not None:
        additional_lines.append('LATHTEST %.3f'%htest)
    if wn_level is not None:
        additional_lines.append('LATWN %.4g'%wn_level)
    par_string = model.as_parfile(
        p_tn=p_tn,p_tne=p_tne,
        additional_lines=additional_lines)
    return par_string

def make_toas(model,dataset,dlogl=10,min_time=2,max_time=56):
    """ Make *actual* TOAs from Fermi data using the model.
    """
    we = dataset.get_weights()
    mjds = dataset.get_mjds()
    ph = model.phase()%1
    logl = np.cumsum(np.log(we*dataset.get_template()(ph)+(1-we)))

    # the algorithm here is somewhat crude but seems good enough:
    # form the smallest possible time intervals satisfying the dlogl
    # requirement.  Combine any intervals in order to satisfy min_time,
    # and discard any intervals exceeding max_time.
    a = np.searchsorted(logl,np.arange(0,int(logl[-1])/dlogl+1)*dlogl)
    mjd0 = mjds[a[:-1]]
    mjd1 = mjds[a[1:]]
    dt = mjd1-mjd0

    return mjds,logl,dt
    #ninterval = int(logl[-1]/dlogl)
    #change_points = np.arange(ninterval+1)*dlogl
    #change_points = np.searchsorted

def logl(model,dataset,use_cache=True):
    we = dataset.get_weights()
    ph = model.phase()%1
    lct = dataset.get_template()
    logl = np.sum(np.log(we*lct(ph,use_cache=use_cache)+(1-we)))
    return logl

def fit_tn_logl(p_tn,self,p0=None,x0=None,debug=False,
        disable_check1=False,disable_check2=False,prior_sigma=None):

    p = self.x0 if (x0 is None) else self.get_p(x0)
    ph,fphi,M,G,H = self.update_cache(p)
    mjoint = self.mjoint
    Gd,Gf = mjoint.get_subspace_G(G)
    Hdd,Hdf,Hff = mjoint.get_subspace_H(H)
    if prior_sigma is not None:
        err = np.diag(np.linalg.inv(Hdd))**0.5
        Hdd += np.diag((err*prior_sigma)**-2)
    alpha = mjoint.tn().get_values()

    # This term isn't used in the power law fit, but is of interest for
    # the GWB, it's the "log likelihood at maximum", which we expect to
    # increase a little bit as more noise is able to be added.
    we = self.weights
    logl0 = np.sum(np.log(we*fphi+(1-we)))

    # remove offset from calculation
    #Hdd = Hdd[1:,1:]
    #Hdf = Hdf[1:]
    #Gd = Gd[1:]

    """
    # add in a 1% prior on offset, F0, F1, etc.
    pvals = mjoint.tm().get_values()
    perrs = np.append([0.01],pvals[1:]*0.01).astype(np.float64)
    # WILL THIS BE A GOOD IDEA FOR F2???  Much more degenerate...
    print(Hdd)
    print((1./perrs)**2)
    Hdd = Hdd + np.diag(perrs**-2)
    print(Hdd)
    """

    p = p_tn

    if p[1] < 0:
        if debug:
            return np.inf,locals()
        else:
            return np.inf
    Ht = mjoint.get_htn(p,zeropad=False)
    # The t1/t2 difference below can be subject to numerical
    # errors if the absolute values become too large (essentially
    # the difference of two very large numbers).  This little
    # check below seems to work pretty well to avoid it.
    t1 = -np.inner(np.diag(Ht),alpha**2)
    if abs(t1) > 1e3*len(alpha):
        if not disable_check1:
            print('Check1 fail')
            if debug:
                return np.inf,locals()
            else:
                return np.inf
        #return 1e100
    # end that little check
    try:
        a1 = HMatrix(Hff + Ht)
        c1 = a1.C()
    except LinAlgError:
        print('LinAlgError1')
        if debug:
            return np.inf,locals()
        else:
            return np.inf
        #return 1e100
    try:
        #a2 = HMatrix(Hdd - Hdf*c1*Hdf.T)
        # CHECK THIS SIGN!!!!
        #a2 = HMatrix(Hdd + Hdf@c1@Hdf.T)
        a2 = HMatrix(Hdd - Hdf@c1@Hdf.T)
        # checked, and it is definitely -ve as above
        c2 = a2.C()
    except LinAlgError:
        print('LinAlgError2')
        if debug:
            return np.inf,locals()
        else:
            return np.inf
    y = Gf-np.diag(Ht)*alpha
    z = Gd-y@c1@Hdf.T
    det_ht = np.sum(np.log(np.diag(Ht)))
    det_a1 = -a1.logdet() # need -ve signs because logdet(C)
    det_a2 = -a2.logdet()
    #t1 = -np.inner(np.diag(Ht),alpha**2)
    t2 = np.inner(np.inner(c1,y),y).item()
    t3 = np.inner(np.inner(c2,z),z).item()
    logl = 0.5*(det_ht - det_a1 - det_a2 + t1 + t2 + t3) + logl0
    #print('%.2f'%det_ht,'%.2f'%det_a1,'%.2f'%det_a2,'%.2f'%t1,'%.2f'%t2,'%.2f'%t3,'%.2f'%logl0)
    # put a check on t2 and t3, too
    if abs(t2) > 1e3*len(alpha):
        if (not disable_check2):
            print('Check2 fail')
            if debug:
                return np.inf,locals()
            else:
                return np.inf
    if abs(t3) > 1e3*len(alpha):
        print('Check3 fail')
        if debug:
            return np.inf,locals()
        else:
            return np.inf
    #if logl > 1e4:
        #print(logl,det_ht,det_a1,det_a2,t1,t2,t3)
    # the only thing this omits is the -0.5*resid*Hw*resid term
    if debug:
        return -logl,locals()
    else:
        return -logl

def dual_fit_tn_logl(p_tn1,p_tn2,self,p0=None,x0=None,debug=False,
        disable_check1=False,disable_check2=False):
    # same as above, but with two TN contributors

    p = self.x0 if (x0 is None) else self.get_p(x0)
    ph,fphi,M,G,H = self.update_cache(p)
    mjoint = self.mjoint
    Gd,Gf = mjoint.get_subspace_G(G)
    Hdd,Hdf,Hff = mjoint.get_subspace_H(H)
    alpha = mjoint.tn().get_values()

    # This term isn't used in the power law fit, but is of interest for
    # the GWB, it's the "log likelihood at maximum", which we expect to
    # increase a little bit as more noise is able to be added.
    we = self.weights
    #logl0 = np.sum(np.log(we*fphi+(1-we)))
    logl0 = 0

    if (p_tn1[1] < 0) or (p_tn2[1] < 0):
        if debug:
            return np.inf,locals()
        else:
            return np.inf
    Ht1 = mjoint.get_htn(p_tn1,zeropad=False,ntm_param=Gd.shape[0])
    Ht2 = mjoint.get_htn(p_tn2,zeropad=False,ntm_param=Gd.shape[0])

    Ct1 = 1./np.diag(Ht1) #NB this is now P/2 in the entries
    Ct2 = 1./np.diag(Ht2)
    Ct = Ct1 + Ct2 # P1/2 + P2/2
    Ht = np.diag(1./Ct) # joint timing noise matrix


    # The t1/t2 difference below can be subject to numerical
    # errors if the absolute values become too large (essentially
    # the difference of two very large numbers).  This little
    # check below seems to work pretty well to avoid it.
    t1 = -np.inner(np.diag(Ht),alpha**2)
    if abs(t1) > 1e3*len(alpha):
        if not disable_check1:
            print('Check1 fail')
            if debug:
                return np.inf,locals()
            else:
                return np.inf
        #return 1e100
    # end that little check
    try:
        a1 = HMatrix(Hff + Ht)
        c1 = a1.C()
    except LinAlgError:
        print('LinAlgError1')
        if debug:
            return np.inf,locals()
        else:
            return np.inf
        #return 1e100
    try:
        #a2 = HMatrix(Hdd - Hdf*c1*Hdf.T)
        # CHECK THIS SIGN!!!!
        #a2 = HMatrix(Hdd + Hdf@c1@Hdf.T)
        a2 = HMatrix(Hdd - Hdf@c1@Hdf.T)
        # checked, and it is definitely -ve as above
        c2 = a2.C()
    except LinAlgError:
        print('LinAlgError2')
        if debug:
            return np.inf,locals()
        else:
            return np.inf
    y = Gf-np.diag(Ht)*alpha
    z = Gd-y@c1@Hdf.T
    det_ht = np.sum(np.log(np.diag(Ht)))
    det_a1 = -a1.logdet() # need -ve signs because logdet(C)
    det_a2 = -a2.logdet()
    #t1 = -np.inner(np.diag(Ht),alpha**2)
    t2 = np.inner(np.inner(c1,y),y).item()
    t3 = np.inner(np.inner(c2,z),z).item()
    logl = 0.5*(det_ht - det_a1 - det_a2 + t1 + t2 + t3) + logl0
    #print('%.2f'%det_ht,'%.2f'%det_a1,'%.2f'%det_a2,'%.2f'%t1,'%.2f'%t2,'%.2f'%t3,'%.2f'%logl0)
    # put a check on t2 and t3, too
    if abs(t2) > 1e3*len(alpha):
        if (not disable_check2):
            print('Check2 fail')
            if debug:
                return np.inf,locals()
            else:
                return np.inf
    if abs(t3) > 1e3*len(alpha):
        print('Check3 fail')
        if debug:
            return np.inf,locals()
        else:
            return np.inf
    #if logl > 1e4:
        #print(logl,det_ht,det_a1,det_a2,t1,t2,t3)
    # the only thing this omits is the -0.5*resid*Hw*resid term
    if debug:
        return -logl,locals()
    else:
        return -logl

    # TODO -- so much time being burned in get_params!!!  Fix...
