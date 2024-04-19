# %%
import numpy as np
import scipy.optimize as op
import pint.logging
import pint.simulation as ps
import pint.models as models
import pint.residuals as pr
import astropy

from pint.models import get_model
from pint.simulation import make_fake_toas_uniform
from pint.fitter import Fitter
from pint.logging import setup as setup_log
from pint.models import parameter as pp
from io import StringIO
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.visualization import quantity_support
from scipy.linalg import cho_solve,cho_factor,solve_triangular,LinAlgError
from loguru import logger as log

pint.logging.setup(level='INFO')
quantity_support
# %%
class TNMatrix(object):
    """ Encapsulate the spectral decomposition form of a TN process."""
    def __init__(self,p,freqs,model,ts,zeropad=None):
        """ NB freqs should be in cycles/year!."""
        self.freqs = freqs
        n = zeropad + 2*len(freqs)
        self._H = np.zeros((n,n),dtype=float)
        x,y = np.diag_indices_from(self._H)
        self._x = x[zeropad:]
        self._y = y[zeropad:]
        self.update_p(p)

        self.model = model
#         self.epoch = model.WAVEEPOCH.value if model.WAVEEPOCH else model.PEPOCH.value
        self.epoch = model.PEPOCH.value
        self.mjds = ts.get_mjds().value
        self.scale = (2/(self.mjds[-1]-self.mjds[0])/365.25)**0.5*self.model.F0.quantity # Units of Hz

    def designmatrix(self):
        ndata = len(self.mjds)
        nharm = len(self.freqs)
        F = np.empty([2*nharm,ndata],dtype=np.float128)
        phase = np.empty_like(self.mjds)
        times = (self.mjds-self.epoch)/365.25 # This is in years now
        for iharm,freq in enumerate(self.freqs):
            phase[:] = (2*np.pi*freq)*times # Units should be freq (1/years) * times (years) to give units of phase
            np.cos(phase,out=F[2*iharm,:])
            np.sin(phase,out=F[2*iharm+1,:])
        F *= self.scale.value # Units for F should now be in Hz
#         F *= self.scale.unit
        return F.T
    
    def eval_pl(self,p,freqs):
        # Make sure that frequencies are in 1/year
        amp,alpha = p
#         amp = amp**2
        return (10**amp)*freqs**-alpha
    
    def get_values(self):
        return self.coeffs
    
    def H(self):
        """ Return inverse covariance matrix."""
        return self._H
    
    def update_p(self,p):
        freqs = self.freqs * 365.25 # *24*3600 # Frequencies are converted into 1/d
#         freqs = self.freqs
        tn_vals = 2./self.eval_pl(p,freqs) 
        self._H[self._x[::2],self._y[::2]] = tn_vals
        self._H[self._x[1::2],self._y[1::2]] = tn_vals
        self._p = p

# %%
def powlaw(x, a, b) :
    return a * np.power(x, b)

def get_amp_ind(residuals,nwaves,res_type=None,show_plot=False):
    if res_type == 'time':
        residuals = residuals.time_resids.to(u.s) # Are units supposed to be in seconds or days
        title=f'FFT of Time Residuals'                  # Days gets me in the estimated values 
    if res_type == 'phase':
        residuals = residuals.phase_resids
        title=f'FFT of Phase Residuals'
    a = np.fft.fft(residuals.astype(np.float64))
    psd = np.abs(a)**2 # The units of the power spectral density should be amp_units**2 / sampling frequency
    freqs = np.fft.fftfreq(len(psd),d=1/(2*len(psd)))
    lim = int(len(psd)/2)
    xdata = freqs[1:nwaves+1]
    ydata = psd[1:nwaves+1]

    popt,pcov = op.curve_fit(powlaw,xdata,ydata,maxfev=1000)
    log.info(f'The Amp is {popt[0]} and the spectral index is {popt[1]}')
    log.info(f'The reporting convection is: logA is {np.log10(popt[0])} and the spectral index is {-popt[1]}')
    if show_plot:

        plt.figure(figsize=(15,10))
        plt.plot(freqs[1:lim], popt[0]*np.power(freqs[1:lim],popt[1]),'--',label='Power-law Fit')
        plt.plot(freqs[1:lim],psd[1:lim],'x')
        plt.title(f'{title}', fontsize = 24)
        plt.xscale('log')
        plt.yscale('log')
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        plt.ylabel('Power Spectral Density (s$^2$)',fontsize = 16)
        plt.xlabel('Sample Frequency',fontsize = 16)
        plt.grid('True')
    
    return [np.log10(popt[0]),popt[1]]


# %% Taken from Matthew's code. Not yet modified
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

# %% 
class RNLikelihoodFit(object):
    def __init__(model,toas,nwaves=None):
        self.model = model
        self.toas = toas
        self.tm = (model.designmatrix(toas)[0] * -model.F0.value).transpose
        self.H_tn = self.generate_H_tn()
        self.F = H_tn.designmatrix()
        self.M = self.joint_designmatrix(model,toas,H_tn=self.H_tn)
        self.nwaves = nwaves
        self.x0 = np.append(np.array([-1.0]),[getattr(self.model,p).value for p in self.model.free_params],self.init_coeffs) # -1.0 is the initial offset value

    def process_dataset(self):
        # get white noise measurements and pulse numbers
        self.toas.compute_pulse_numbers(self.model)
        self._pn = self.toas.get_pulse_numbers()
        errs = self.toas.get_measurement_errors()
        self.h_vec = 1./errs**2
        self.Hw = np.diag(self.h_vec)

    def estimate_p(self):
        '''
        Returns an estimate of the logA and gamma based on the toas and model prefit residuals
        '''
        resids = pr.Residuals(self.toas,self.model)
        self.p = get_amp_ind(resids,nwaves=nwaves,res_type='time')
    
    def coeffs_lsq(self):
        '''
        Returns a starting estimate of red noise coefficients using Least Squares
        '''
        resids = pr.Residuals(self.toas,self.model)
        phase_res = resids.phase_resids.astype(np.float64)
        self.init_coeffs = np.linalg.lstsq(self.F,phase_res,rcond=None)
    
    def joint_designmatrix(model,toas,H_tn=None):
        D, params, units = model.designmatrix(toas)
        D *= -model.F0.value
        D = D.transpose()

        if H_tn:
            F = H_tn.designmatrix()
            freqs = H_tn.freqs
            output = np.empty((D.shape[0]+F.shape[0],D.shape[1]),dtype=np.float64)
            output[:D.shape[0]]=D
            output[D.shape[0]:]=F
            return output, F.T, freqs
        return D, None, None
    
    def generate_H_tn(self):
        mjds = self.toas.get_mjds() # mjds is in units day
        tobs = (mjds[-1]-mjds[0]).to_value(u.yr) #Tobs is in units year
        self.freqs = np.linspace(1,self.nwaves,self.nwaves)/tobs * 1/u.yr # units are 1/yr
        self.n_params = len(self.model.free_params)
        self.estimate_p()
        self.coeffs_lsq()
        H_tn = TNMatrix(self.p, self.freqs,self.model,self.toas,self.n_params)
        return H_tn
    
    def get_phase(self):
        ph = self.model.phase(self.toas)
        ph = ph[0]+ph[1]
        rn_ph = np.inner(self.F,self.init_coeffs)
        return ph+rn_ph
    
    def data_goodies(self):
        ''' This is slightly modified from the Gaussian Likelihood fit in Matthew's code'''
        ph = self.get_phase()
        fph = self._pn - ph # residuals
        M = self.M
        # G is design matrix * error matrix * residuals
        G = np.inner(M,fph*self.h_vec)
        # this is "Hlambda" elsewhere
        H = np.asarray((M*self.h_vec)@M.T)
        return ph,fph,M,G,H
    

# %%
# %%
m = get_model(
   StringIO(
        """
        RAJ    		05:00:00
        DECJ   		20:00:00
        F0     		300     1 
        F1     		-1e-14  1
        PEPOCH 		58500
        #DM     		15
        #EFAC tel gbt 	1.3
        TNRedAmp 	-9.5
        TNRedGam 	4.0
        TNRedC 		30
        """	
    )
)
t = make_fake_toas_uniform(
    57000, 
    60000, 
    100, 
    model=m, 
    error=1 * u.ms, 
    #freq=u.Quantity([1400*u.MHz, 2500*u.MHz]), 
    add_noise=True, 
    add_correlated_noise=m.has_correlated_errors
)
# %%
residuals = pr.Residuals(t,m)
# %%
[getattr(m,p).value for p in m.free_params]
# %%
