from __future__ import print_function,division

import numpy as np
from scipy.linalg import cho_solve,cho_factor,cholesky,solve_triangular,LinAlgError
from scipy.special import gamma,kv
from scipy.stats import multinomial

def eval_pl(p,freqs):
    amp,alpha = p
    return (10**amp)*freqs**-alpha
    

def get_random_tn(nbin,index=-2):
    """ Return a band-limited realization of timing noise with the indicated
        spectral index.  Zero-frequency power is set to 0.

        By construction, the resulting time series will obey periodic
        boundary conditions.  Therefore, it is not necessarily a perfect
        realization of timing noise as realized in real data.  Thus:

        For a version with out-of-band, low-frequency components, consider
        simply using more bins (say, 4x) and then truncating the resulting
        time series.
    """
    odd = nbin%2 != 0
    if odd:
        nbin += 1
    a = np.empty(nbin//2+1,dtype=np.complex128)
    real_coeffs = np.random.randn(nbin//2+1)
    freqs = np.fft.rfftfreq(nbin)
    real_coeffs *= freqs**(index*0.5)
    imag_coeffs = np.random.randn(nbin//2+1)
    imag_coeffs *= freqs**(index*0.5)
    a.real = real_coeffs
    a.imag = imag_coeffs
    a[0] = 0
    x = np.fft.irfft(a)
    if odd:
        nbin -= 1
    return (x * 1./np.abs(x).max())[:nbin]

def get_random_tn_corner(p,nbin):
    """ Return a band-limited realization of timing noise with the indicated
        spectral index.  Zero-frequency power is set to 0.

        By construction, the resulting time series will obey periodic
        boundary conditions.  Therefore, it is not necessarily a perfect
        realization of timing noise as realized in real data.  Thus:

        For a version with out-of-band, low-frequency components, consider
        simply using more bins (say, 10x) and then truncating the resulting
        time series.  Or use it for extra realizations for MCMC.

        Normalization of power: god this one is tough.  But, applying
        Wiener Khincin using something white, I get the integral of the PSD
        gives 1/2 if A=1, and the variance of the signal is 1.  
        And actually, I think this is right, because K would be for fully
        complex series and integrate over all freqs.
        So...  I think this is correct as is.  But maybe off by 2.

        NB that this means that the level of noise (as realized by auto-
        correlation) in the time domain depends on the length of the
        simulation!  But this is as it should be.  (WK with white noise
        is unbounded, so if you use this to make band-limited white noise,
        it will grow with the length of hte data.)

        Finally, on units: this current version with uniform sampling
        assumes that one bin is one unit, here one day.

    Parameters
    ----------
    p : tuple of floats (a,alpha,fc)
        The parameters for the power law.
        a = log_10 power density amplitude in day^3
        alpha = spectral index of power law; convention is alpha >= 0
        fc = log_10 corner frequency  in 1/day

    """

    amp,alpha,fc = p
    amp = 10**amp
    fc = 10**fc
    
    odd = nbin%2 != 0
    if odd:
        nbin += 1
    freqs = np.fft.rfftfreq(nbin) # here we are assuming 1-day sampling!
    psd = amp*(1+(freqs/fc)**2)**(-0.5*alpha)
    psd[0] = 0 # mean subtract
    # WK check -- below line should equal variance of time-domain output
    # to within sample realization variance.  Factor of 2 is because WK
    # should be summed over negative frequencies too.  (And presumably
    # have a complex time series.)
    wk = (psd*(freqs[1]-freqs[0])).sum()*2
    scale = (psd/2)**0.5 # CHECK FACTOR OF 2

    a = np.empty(nbin//2+1,dtype=np.complex128)
    a.real = np.random.randn(nbin//2+1)*scale
    a.imag = np.random.randn(nbin//2+1)*scale
    # NB that FFT scales by sqrt(N) for forward and 1/sqrt(N) for ifft
    # Therefore the scaling below seems to keep WK satisfied.  The factor
    # of sqrt(2) is not clear but not unexpected...
    x = np.fft.irfft(a)
    if odd:
        nbin -= 1
    x = x[:nbin]*(len(a)*2)**0.5
    return x,wk,psd,a

def get_random_tn2(p,nbin,tbin=1./365,fcut_lo=None,fcut_hi=None):
    """ Return a band-limited realization of timing noise with the indicated
        spectral index.  Zero-frequency power is set to 0.

        By construction, the resulting time series will obey periodic
        boundary conditions.  Therefore, it is not necessarily a perfect
        realization of timing noise as realized in real data.  Thus:

        For a version with out-of-band, low-frequency components, consider
        simply using more bins (say, 10x) and then truncating the resulting
        time series.  Or use it for extra realizations for MCMC.

        Normalization convention: this is determined such that the 
        Wiener-Khinchin relation holds.  Specifically, consider that you
        use an input power spectrum S(f) = S0, i.e. white noise.  Then
        Then the resulting time series variance is

        Var(x[t]) = 2 int_0^fmax S(f) = 2*S0*fmax = S0/tbin.

        On the other hand, if the input power spectrum is a "delta",
        Var(x[t]) = 2 int_0^fmax S(f) = 2*S0*delta_f = 2*S0/T.

        Both cases work with the normalization applied here.
        ***
        To recover the input power spectrum from the output time series,
        |FFT(x)|^2/len(x)*tbin.
        ***
        This has been verified.

        One consequence of this is that the total power in the output scales
        with its length.  In particular,
        (1) For a red spectrum, the total power output doesn't depend on
            sampling, only on the total length in time units.
        (2) For a white spectrum, the total power *does* depend on both the
            sampling and the length of the data, since finer time scales
            probe "new" power.

        Have done some sanity checks, e.g. calling with p=[-16,0] for 10yr
        with monthly sampling (tbin=1/12) yields a typical measurement
        precision of 30ns, which is in the right ballpark for J1909-3744
        given the frequent sampling and that the white noise should be
        below the limit of -15.

        On the other hand, if simulating a -15 process with similar
        parameters, the power goes up to microseconds, but this is because
        the low-frequency modes drive it, whereas of course these are
        smoothed out in the real data.

        For generating realistic timing noise, it simply suffices to aim
        for a typical TN amplitude on the timescale of your data.  So say
        you want ~1s residuals after subtracting F0/F1, and your data span
        is 10 years, and the slope is -6, then the amplitude should be
        (10yr/1yr)^-6 = 1e-6.

    Parameters
    ----------
    p : tuple of floats (a,alpha)
        The parameters for the power law.
        a = log_10 power density amplitude in s^2*yr
        alpha = spectral index of power law; convention is alpha >= 0
    """

    amp,alpha = p
    amp = 10**amp
    
    odd = nbin%2 != 0
    if odd:
        nbin += 1
    # because bins are in units of 1yr, so are frequencies, so by fiat the
    # normalization here is A = S(f=1/yr).
    freqs = np.fft.rfftfreq(nbin)/tbin
    freqs[0] = 1e-10 # just avoid numerical whinging
    psd = amp*freqs**(-alpha)
    if fcut_lo is not None:
        psd[freqs <= fcut_lo] = 0
    if fcut_hi is not None:
        psd[freqs > fcut_hi] = 0
        #psd *= np.exp(-(freqs/fcut_hi)**4)
    psd[0] = 0 # mean subtract
    # WK check -- below line should equal variance of time-domain output
    # to within sample realization variance.  Factor of 2 is because WK
    # should be summed over negative frequencies too.
    #total_time =  nbin*tbin
    wk = (psd*(freqs[2]-freqs[1])).sum()*2
    scale = psd**0.5/2

    a = np.empty(nbin//2+1,dtype=np.complex128)
    a.real = np.random.randn(nbin//2+1)*scale
    a.imag = np.random.randn(nbin//2+1)*scale
    # NB that FFT scales by sqrt(N) for forward and 1/sqrt(N) for ifft,
    # and we also need the normalization for the time span, so we end up
    # with a linear scaling in nbin.
    x = np.fft.irfft(a)
    if odd:
        nbin -= 1
    #total_time =  nbin*tbin
    #x = x[:nbin]*(len(a)/tbin*total_time)**0.5
    x = x[:nbin]*(nbin/tbin)**0.5
    return x,wk,psd,freqs

def subharmonic_coeffs(sig,nharms):
    ph = 2*np.pi*np.linspace(0,1,len(sig))
    H = np.empty((nharms*2,len(sig)))
    M = np.empty(nharms*2)
    M2 = np.empty((nharms*2,nharms*2))
    for i in range(nharms):
        H[i] = np.cos(ph*0.5**(i+1))
        H[nharms+i] = np.sin(ph*0.5**(i+1))
        M[i] = (H[i]*sig).sum()
        M[nharms+i] = (H[nharms+i]*sig).sum()
    for i in range(2*nharms):
        for j in range(i,2*nharms):
            M2[i,j] = (H[i]*H[j]).sum()
            M2[j,i] = M2[i,j]
    C = cho_factor(M2,lower=True)
    coeffs = cho_solve(C,M)
    predicted_sig = np.einsum('i,ij->j',coeffs,H)
    return coeffs,predicted_sig

def harmonic_coeffs(sig,nharms):
    ph = 2*np.pi*np.linspace(0,1,len(sig))
    H = np.empty((nharms*2,len(sig)))
    M = np.empty(nharms*2)
    M2 = np.empty((nharms*2,nharms*2))
    for i in range(nharms):
        H[i] = np.cos(ph*(i+1))
        H[nharms+i] = np.sin(ph*(i+1))
        M[i] = (H[i]*sig).sum()
        M[nharms+i] = (H[nharms+i]*sig).sum()
    for i in range(2*nharms):
        for j in range(i,2*nharms):
            M2[i,j] = (H[i]*H[j]).sum()
            M2[j,i] = M2[i,j]
    C = cho_factor(M2,lower=True)
    coeffs = cho_solve(C,M)
    predicted_sig = np.einsum('i,ij->j',coeffs,H)
    return coeffs,predicted_sig


def resample_phase(template,we,mjd_bt,epoch=None,F0=1,F1=1e-15):
    """ Given a template, a set of weights, and a set of barycentering
        times, generate a new set of times that follow a different model,
        taken here to be a very simple quadratic (in phase) spindown.
    """
    dom = np.linspace(0,1,1001)
    edom = 0.5*(dom[:-1]+dom[1:])
    cod = template(dom)
    F = np.cumsum(cod)
    F *= 1./F[-1]


    src_mask = np.random.rand(len(we)) <= we
    nsrc = src_mask.sum()
    nbkg = len(we)-nsrc
    new_phas = np.empty_like(we)
    new_phas[src_mask] = edom[np.searchsorted(F,np.random.rand(nsrc))-1]
    new_phas[~src_mask] = np.random.rand(nbkg)

    # now, need to assign new times for all of the photons based on the phase model
    if epoch is None:
        epoch = mjd_bt.mean()
    pred_phase = F0*(mjd_bt-epoch)*86400 + 0.5*F1*(mjd_bt-epoch)**2*86400**2
    dt = ((new_phas-pred_phase)%1)/F0

    mjd_bt = mjd_bt + dt/86400
    ph = np.mod(F0*(mjd_bt-epoch)*86400 + 0.5*F1*(mjd_bt-epoch)**2*86400**2,1)
    return mjd_bt,ph


def pre_whitening_demo(idx=-5):
    """ Demonstration of controlling spectral leakage.

    Use finite difference pre-whitening and post darkening.

    This also suggests a potential algorithmic prescription: generally,
    the best filtering will still have non-white noise, so the rms of
    the PSD will be minimum at one step past optimal.  Therefore simply
    detect this.
    """


    # demonstration of controlling spectral leakage
    # generate signal and truncate to 1/10 the length
    sig = get_random_tn(4096*10,index=idx)[:4096]
    # signal follows f^-idx, so should take floor(idx/2) rounds of finite 
    # difference pre-whitening to get the spectrum correct
    d0 = sig
    d1 = d0[1:]-d0[:-1]
    d2 = d1[1:]-d1[:-1]
    d3 = d2[1:]-d2[:-1]
    d4 = d3[1:]-d3[:-1]

    import pylab as pl
    pl.clf()
    # draw the truth
    dom = np.fft.rfftfreq(len(d0))[1:]
    pl.loglog(dom,(dom/dom[0])**idx,label='Truth',color='k',lw=2)
    rms = []
    for i,d in enumerate([d0,d1,d2,d3,d4]):
            dom = np.fft.rfftfreq(len(d))[1:]
            psd = np.abs(np.fft.rfft(d)[1:])**2
            rms.append((psd**0.5).std())
            psd *= 1./psd[0]
            scale = dom**-(2*i)
            scale *= 1./scale[0]
            pl.loglog(dom,psd*scale,label='Pre-whitening=%d'%i,alpha=0.5)
    pl.legend(loc='lower left')
    a = np.argmin(rms)
    print('Best guess for pre-whitening: %d steps.'%(a-1))

# Matern covariance for corner-freq power law.
def covariance(p,lags):
    """ Return the covariance for a Matern kernel mapping to a power law
        with corner frequency.

    Parameters
    ----------
    p : tuple of floats (a,alpha,fc)
        The parameters for the power law.
        a = log_10 (power density amplitude)
            can be arbitrary units, but suggest 1/unit(lag)^2
        alpha = spectral index of power law; convention is alpha >= 0
        fc = log_10 (corner frequency (units of lag))
    lags : array (float)
        The time difference between observations relative to the first one.

    This confirmed to follow the formula in Delphine's paper, with the
    exception that her corner frequency is perhaps an angular frequency so
    there needs to be some 2pi action.

    NB kv blows up near 0, normalized by multiplication, sso need to use 
    asymptotic result.
    """


    a,alpha,fc = p
    amp = 10**a
    fc = 10**fc
    q = np.abs((2*np.pi*fc)*np.asarray(lags,dtype=float))
    e = float(0.5*(alpha-1))
    # original scaling had a sqrt(2*np.pi), I suspect a guesstimate
    # I am still off on the power normalization.  Empirically checking a
    # range of indices suggests it's ~5/2, so that's added in by hand now.
    # it diverges somewhat when alpha-->2 (15%), so may be worth treating
    # the alpha=2 case more explicitly
    coeff = amp*fc*2**(1-0.5*alpha)/gamma(float(0.5*alpha))*(5./2)
    #coeff = 10**a*fc*2**(1-0.5*alpha)/gamma(float(0.5*alpha))
    #print(coeff,e,2**(1-0.5*alpha))
    rvals = np.empty_like(q)
    rvals[q>0] = q[q>0]**e*kv(-e,q[q>0])
    rvals[q==0] = 2**(e-1)*gamma(e)
    return rvals*coeff

def make_design_matrix(ntay,freqs,times,psd_scale=None,window=False,
        orthogonalize=False):
    """
    freqs -- frequencies to fit in 1/yr
    times -- times of each sample (yr)
    """
    ndata = len(times)
    nharm = len(freqs)
    nparm = 2*nharm + ntay
    M = np.empty([nparm,ndata],dtype=np.float128)
    M[0,:] = 1. # mean
    DT = times[-1]-times[0]
    scale = 2./DT
    dx = times*scale # numerical scaling to make Taylor similar to harms
    for i in range(1,ntay):
        M[i,:] = dx/i*M[i-1,:]
        #M[i,:] -= M[i,:].mean()
    phase = np.empty_like(times)
    for iharm in range(nharm):
        phase[:] = (2*np.pi)*times*freqs[iharm]
        np.cos(phase,out=M[ntay+2*iharm,:])
        np.sin(phase,out=M[ntay+2*iharm+1,:])
    if psd_scale is not None:
        M[ntay:] *= psd_scale
    if orthogonalize:
        Mold = M.copy()
        for i in range(nparm-1):
            amp = np.sum(M[i]*M[i+1:],axis=1)
            M[i+1:] -= amp[:,None]*M[i]/np.sum(M[i]**2)
        M[ntay:] *= (Mold[ntay:].std(axis=1)/M[ntay:].std(axis=1))[:,None]
    if window:
        from scipy.signal import get_window
        from scipy.signal import windows
        x = get_window('blackmanharris',ndata)[None,:]
        #x = windows.chebwin(ndata,100,sym=False)[None,:]
        #x *= 1./x.std()**2
        M[ntay:] *= 1./x
    return M

class DataMaker(object):
    """ Simulate time series.
    
    TODO -- ultimately are going to want to specify the noise model at
    instantiation, which will let us cache the PSD stuff.
    
    To handle uneven data sets, oversample by a factor, but don't simulate
    noise higher than the intrinsic data binning, which is generally taken
    to be 0.5/day, a reasonable high-frequency cutoff for physical
    processes."""

    def __init__(self,p0,tbin=1./365,typical_cadence=21,nsamp=174,
            oversample=10,exact_spacing=False,fcut_hi=None):
        """
        tbin -- size of fundamental bin in years
        typical_cadence -- typical length between observations, in bins
        nsamp -- total number of observations
        oversample -- when simulating, oversample by this amount (relative
            to the fundamental bin size)
        """

        self.p0 = p0
        if not exact_spacing:
            p = np.ones(nsamp-1)*(1./(nsamp-1))
            gaps = multinomial.rvs(typical_cadence*(nsamp-1)*oversample,p)*(1./oversample)
        else:
            gaps = np.ones(nsamp-1)*typical_cadence
        # NB internal times are in years
        self.internal_times = np.cumsum(np.append(0,gaps*tbin))
        self.tbin = tbin
        self.cadence = typical_cadence
        self.nsamp = nsamp
        self.oversample = oversample
        self.indices = np.cumsum(np.append(0,np.round(gaps*oversample).astype(int)))
        self.fcut_hi = fcut_hi
        self._sim_samp = int(128**0.5*self.indices[-1])
        self._last_idx = self._sim_samp

    def _sim_new(self):
        a1,wk,psd_sim,a = get_random_tn2(
                p0,sim_samp,tbin=self.tbin/self.oversample,
                fcut_hi=self.fcut_hi or 0.5/self.tbin)

    def get_data(self):
        if self._last_idx > (self._sim_samp - self.indices[-1]):
            a1,wk,psd_sim,a = get_random_tn2(
                    self.p0,self._sim_samp,tbin=self.tbin/self.oversample,
                    fcut_hi=self.fcut_hi or 0.5/self.tbin)
            self._a1 = a1
            self._last_idx = 0
        sig = self._a1[self._last_idx + self.indices]
        self._last_idx += self.indices[-1]
        return self.internal_times.copy(),sig

    def get_autocorrelation(self,max_lags=100):
        """ Return the autocorrelation for the current time series.

        Uses a slow, explicit calculation for ease of understanding.
        """
        rvals = np.empty(max_lags)
        for i in range(max_lags):
            ndat = len(self._a1)-i
            rvals[i] = np.mean(self._a1[:ndat]*self._a1[i:ndat+i])
        return self.tbin*np.arange(max_lags),rvals

def make_H(M,ntay,wn_sig):
    """ Convenience routine to combine measurement error and design matrix.
    
        wn_sig can either be a scalar (same for every measurement) or a
        vector (heterogeneous errors) but cannot be a full coviarance
        matrix (yet)."""
    nparm = M.shape[0]
    M = M * (1./np.atleast_1d(wn_sig)[None,:])
    H = np.zeros((nparm,nparm),dtype=np.float128)
    for iparm in range(nparm):
        for jparm in range(iparm,nparm):
            t1 = (M[iparm]*M[jparm]).sum()#/wn_sig**2
            H[iparm,jparm] = H[jparm,iparm] = t1 
    return H

