from copy import deepcopy

from matplotlib.ticker import FuncFormatter,LinearLocator,FixedLocator
from pint.templates import lcfitters
import pylab as pl
import numpy as np

import utils
import tn_utils

class PlottingData(object):

    def __init__(self,model,dataset,ph=None,we=None):
        self.model = model
        self.dataset = dataset
        self.ph = model.modphase() if ph is None else ph
        self.we = dataset.get_weights() if we is None else we

def plot_psd(plotting_data,popt,fignum=1,wn_level=None):
    pl.figure(fignum); pl.clf()
    ax = pl.gca()
    model = plotting_data.model
    allow_f2 = model.get_allow_f2()
    freqs,psd = model.get_psd()
    psd_mod = tn_utils.eval_pl(popt,freqs)
    ax.loglog(freqs,psd,marker='.')
    ax.loglog(freqs,psd_mod,lw=2,label='logA=%.2f, $\gamma$=%.2f, F2=%d'%(
        popt[0],popt[1],allow_f2))
    if wn_level is not None:
        pl.axhline(wn_level,label='logWN=%.2f'%(np.log10(wn_level)),
                color='C3')
    ax.set_xlabel('Frequency (cycles yr$^{-1}$)')
    ax.set_ylabel('Power Spectral Density (s$^2$ yr$^{-1}$)')
    ax.legend(loc='upper right')
    ax.axis([0.07,12,8e-12,2e3])

def plot_hist2d(plotting_data,logl=0,fignum=2):
    pl.figure(fignum); pl.clf()
    model = plotting_data.model
    dataset = plotting_data.dataset
    bins = [100,25]
    if logl > 1000:
        bins = [100,25]
    if logl > 2000:
        bins = [200,50]
    if logl > 5000:
        bins = [400,100]
    mjds = dataset.get_mjds()
    pl.hist2d(mjds,plotting_data.ph,weights=plotting_data.we,bins=bins)
    glitch_epochs = utils.get_glitch_epochs(model.pint_model())
    for glep in glitch_epochs:
        pl.axvline(glep,ls='--',color='C1')
    pl.xlabel('MJD')
    pl.ylabel('Pulse Phase')
    pl.axis([mjds[0],mjds[-1],0,1])
    pl.tight_layout()

def plot_pulse_profile(plotting_data,logl=0,fignum=3):
    lct = deepcopy(plotting_data.dataset.get_template())
    lcf = lcfitters.LCFitter(lct,plotting_data.ph,weights=plotting_data.we)
    nbins = 50
    if logl > 500:
        nbins = 100
    if logl > 1000:
        nbins = 200
    if logl > 5000:
        nbins = 400
    pl.figure(fignum); pl.clf()
    ax = pl.gca()
    lcf.plot(axes=ax,nbins=nbins)
    pl.tight_layout()
    return lcf,nbins

def plot_2d_like(plotting_data,logl=0,fignum=4):
    model = plotting_data.model
    dataset = plotting_data.dataset
    ph = plotting_data.ph
    we = plotting_data.we
    lct = dataset.get_template()
    ti = dataset.get_mjds()
    glepochs = utils.get_glitch_epochs(model.pint_model())

    # This can be calculated more specifically with likelihood drops
    if logl == 0:
        tstep = 14
        window = 2
    else:
        nstep = logl/5 # symmetric window, likelihood drop of 20 total
        tstep = max(1,(ti[-1]-ti[0])/nstep)
        window = 2

    window = int(round(window))
    tstep = int(round(tstep))

    plot_2d_like_base(lct,ph,we,ti,window=window,tstep=tstep,clip=20,
            phi_range=None,fignum=fignum,glepochs=glepochs,symmetric=True,
            title='Window=%d, Tstep=%d'%(window,tstep))

def plot_toa_like(plotting_data,fignum=5):
    model = plotting_data.model
    dataset = plotting_data.dataset
    ph = plotting_data.ph
    we = plotting_data.we
    lct = dataset.get_template()
    ti = dataset.get_mjds()
    glepochs = utils.get_glitch_epochs(model.pint_model())

    phi_range = 0.1
    nphi = 200
    tedges,phdom,logls = plot_2d_like_base(lct,ph,we,ti,window=1,tstep=182,
            nphi=nphi,phi_range=phi_range,symmetric=False,get_logls=True)

    pl.figure(fignum);
    pl.clf()
    nphi = logls.shape[-1]
    dt = tedges[1:]-tedges[:-1]
    tcens = 0.5*(tedges[1:]+tedges[:-1])
    i0 = nphi//2
    assert(phdom[i0] == 0)
    lss = ['-','--','-.']
    for i in range(logls.shape[0]):
        # trim off last bin if not long enough
        if dt[i] < 150:
            continue
        ls = lss[i//10]
        color = 'C%d'%(i%10)
        ll = logls[i]
        cod = ll.max()-ll
        if cod[i0] < 1:
            alpha = 0.3
        else:
            alpha = 1.0
        #a = np.argmin(cod)
        
        pl.plot(phdom,logls[i].max()-logls[i],label='%02d'%i,alpha=alpha,
                ls=ls,color=color)
    pl.axis([-0.5*phi_range,0.5*phi_range,-1,10])
    pl.legend(ncol=4)


def plot_2d_like_base(lct,ph,we,mjds,tmin=None,tmax=None,window=4,tstep=1,
        nphi=100,phi_range=None,clip=None,symmetric=True,
        fignum=40,title=None,output=None,get_rvals=False, glepochs=[],
        get_logls=False):
    """ Construct a phase versus time plot by sliding the timing template
        along the data at intervals of tstep and computing, along the phase
        axis, the likelihood for a given phase shift.
        
        The likelihoods of "window" tstep intevals are then summed
        (smoothed)."""

    # TODO -- factor the profile likelihood into lcfitters
    ph = np.mod(ph,1)
    if phi_range is None:
        phdom = np.linspace(-0.5,0.5,nphi+1)[:-1]
        phi_range = 1
    else:
        phdom = np.linspace(-0.5*phi_range,0.5*phi_range,nphi+1)[:-1]
    assert(nphi == len(phdom))
    ti = mjds
    t0 = tmin or ti.min()
    t1 = tmax or ti.max()
    dt = float(tstep)
    nbins = int(float(t1-t0)/dt)+1
    tedges = t0 + np.arange(0,nbins+1)*dt

    #lct.set_cache_properties(ncache=nphi*10)

    def loglikelihood(myphi,mywe):
        logls = np.empty(nphi)
        for ix,x in enumerate(phdom):
            vals = lct((myphi+x)%1,use_cache=True)
            logls[ix] = np.log(1+mywe*(vals-1)).sum()
        return logls

    logls = np.empty([nbins,nphi])
    for i in range(nbins):
        mask = (ti >= tedges[i]) & (ti < tedges[i+1])
        logls[i,:] = loglikelihood(ph[mask],we[mask])

    if get_logls:
        return tedges,phdom,logls

    smooth_logls = np.empty_like(logls)
    print(window,smooth_logls.shape)
    for i in range(nbins):
        smooth_logls[i,:] = logls[i:i+window,:].sum(axis=0)
    if symmetric:
        smooth_logls = smooth_logls[::-1]
        logls = logls[::-1,:]
        for i in range(nbins):
            # don't double count current likelihood
            smooth_logls[i,:] += logls[i+1:i+window,:].sum(axis=0)
        smooth_logls = smooth_logls[::-1]
        logls = logls[::-1,:]

    tcens = t0 + np.arange(0,nbins+1)*dt + 0.5*dt
    #return tcens,smooth_logls

    # for plotting, it works exceedingly well to normalize
    # each line to the maximum of the likelihood
    normed = np.asarray([x-x.max() for x in smooth_logls])
    #normed = np.roll(normed,nphi//2,axis=1) # put fiducial phase at center
    xscale = max(1,int(round(1000./normed.shape[0])))
    yscale = max(1,int(round(1000./normed.shape[1])))
    #normed = scale_image(normed,xscale=xscale,yscale=yscale)
    if clip is not None:
        clip = -abs(clip)
        normed[normed < clip] = clip
    pl.figure(fignum,(5,8)); pl.clf()
    ax = pl.gca()
    #twin = ax.twinx()
    ax.imshow(normed,interpolation='nearest',aspect='auto',origin='lower')
    ax.axvline(0.5*normed.shape[1],color='white')

    xax = ax.get_xaxis()
    xtick_func = lambda x,p: '%.2f'%(phi_range*(float(x)-phi_range*0.5)/normed.shape[1]-0.5*phi_range)
    xax.set_major_formatter(FuncFormatter(xtick_func))
    xax.set_major_locator(LinearLocator(6))
    ax.set_xlabel('Relative Phase')

    yax = ax.get_yaxis()
    dtdy = (tedges[-1]-tedges[0])/normed.shape[0]
    # TODO -- this seems to be rounding down on the last fig...
    #ytick_func = lambda y,p: '%.3f'%(float(y*dtdy+t0)/1e8)
    ytick_func = lambda y,p: '%d'%(int(round(float(y*dtdy+t0))))
    yax.set_major_formatter(FuncFormatter(ytick_func))
    yax.set_major_locator(LinearLocator(6))
    ax.set_ylabel('MJD')
    ax.axis([0,normed.shape[1],0,normed.shape[0]])

    for glepoch in glepochs:
        ax.axhline((glepoch-t0)/dtdy,color='white',ls='--',lw=2)

    pl.tight_layout()

    if title is not None:
        pl.title(title)

    if output is not None:
        pl.savefig(output)

    #ytwin = twin.get_yaxis()
    #ytick_func = lambda y,p: '%d'%(round(met2mjd(y*dt+t0)))
    #ytwin.set_major_formatter(FuncFormatter(ytick_func))
    #ytwin.set_major_locator(LinearLocator(6))
    #twin.set_ylabel('MJD')

    #ax.set_aspect('equal')
    #twin.set_aspect('equal')
    if get_rvals:
        return tcens,logls,smooth_logls

