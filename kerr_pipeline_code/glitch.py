import argparse
from copy import deepcopy
import glob
import sys

import numpy as np

import common_paper_code as common
import utils

def fast_z2m(phases,weights,m=2):
    """ Use sum/difference relations to evaluate related harmonics.

    If we want this to be even faster, instantiate a class so we don't
    need to allocate any new memory.
    """
    c = np.cos(phases*(2.*np.pi))
    s = np.sin(phases*(2.*np.pi))
    cc = c.copy()
    ss = s.copy()
    z = np.empty(m)
    z[0] = np.sum(weights*cc)**2  + np.sum(weights*ss)**2
    for i in range(1,m):
        t = cc*c - ss*s
        ss = ss*c + cc*s
        cc = t
        z[i] = np.sum(weights*cc)**2  + np.sum(weights*ss)**2
    #return (2./(weights**2).sum())*np.cumsum(z)
    return (2./(weights**2).sum())*np.sum(z)

def fast_hm(phases,weights,m=20):
    """ Use sum/difference relations to evaluate related harmonics.

    If we want this to be even faster, instantiate a class so we don't
    need to allocate any new memory.
    """
    c = np.cos(phases*(2.*np.pi))
    s = np.sin(phases*(2.*np.pi))
    cc = c.copy()
    ss = s.copy()
    z = np.empty(m)
    z[0] = np.sum(weights*cc)**2  + np.sum(weights*ss)**2
    for i in range(1,m):
        t = cc*c - ss*s
        ss = ss*c + cc*s
        cc = t
        z[i] = np.sum(weights*cc)**2  + np.sum(weights*ss)**2
    ts = (2./(weights**2).sum())*np.cumsum(z) - 4.0*np.arange(0,m)
    return ts.max()

class FastZ2M(object):

    def __init__(self,weights):
        self.weights = weights
        self.norm = 2./(weights**2).sum()
        self._wmem1 = np.empty(len(weights))
        self._wmem2 = np.empty(len(weights))
        self._wmem3 = np.empty(len(weights))
        self._wmem4 = np.empty(len(weights))
        self._wmem5 = np.empty(len(weights))

    def __call__(self,phases,m=2):
        phases = np.multiply(2*np.pi,phases,out=self._wmem3)
        c = np.cos(phases,out=self._wmem1)
        s = np.sin(phases,out=self._wmem2)
        self._wmem3[:] = c
        self._wmem4[:] = s
        cc = self._wmem3
        ss = self._wmem4
        t = self._wmem5
        z = np.empty(m)
        weights = self.weights
        z[0] = np.sum(weights*cc)**2  + np.sum(weights*ss)**2
        for i in range(1,m):
            t = np.subtract(cc*c,ss*s,out=t)
            ss = np.add(ss*c,cc*s,out=ss)
            cc[:] = t
            z[i] = np.sum(weights*cc)**2  + np.sum(weights*ss)**2
        #return (2./(weights**2).sum())*np.cumsum(z)
        return self.norm*np.sum(z)

def f0_f1_zgrid(dtbs,phases,weights,f0_tol=0.1,f1_tol=0.1,
        f0_min=0,f0_max=2e-5,
        f1_min=-1e-13,f1_max=1e-13,m=4):

    dt_sec = dtbs * 86400 
    time_range = dt_sec.max()-dt_sec.min()
    f0_scale = f0_tol/time_range
    f1_scale = f1_tol/time_range**2
    nf0 = int((f0_max-f0_min)/f0_scale)
    nf1 = int((f1_max-f1_min)/f1_scale)
    nf0 += int(nf0&1==0)
    nf1 += int(nf1&1==0)
    print(nf0,nf1)

    dt1 = dt_sec
    dt2 = dt_sec**2

    df0s = np.linspace(f0_min,f0_max,nf0)
    df1s = np.linspace(f1_min,f1_max,nf1)
    zs = np.empty([len(df0s),len(df1s)])
    z2m = FastZ2M(weights)
    for if1,f1 in enumerate(df1s):
        ph = phases + f1*dt2
        for if0,f0 in enumerate(df0s):
            zs[if0,if1] = z2m(f0*dt1 + ph,m=m)

    a = np.argmax(zs)
    f0_idx = int(a/zs.shape[1])
    f1_idx = a - f0_idx*zs.shape[1]
    best_f0 = df0s[f0_idx]
    best_f1 = df1s[f1_idx]
    best_z = zs[f0_idx,f1_idx]
    best_phase = best_f0*dt1 + best_f1*dt2

    return best_z,best_f0,best_f1,best_phase,df0s,df1s

def solve_glitch(model,dataset,epoch,prec_tol=0.02,target_dw=30,min_time=50,
        max_recurse=2,f0_min=0,f0_max=1e-4,f1_min=-8e-13,f1_max=8e-13,
        f1_only=False,epoch_tol=1,skip_refine=False):
    """
    epoch_tol [days] -- Tolerance for best-fit to agree with guess epoch.
    """
    # first, determine an appropriate amount of time over which to fit
    # the pre-glitch parameters; this would/could be the longest time
    # over which the solution is stable with F0/F1
    t0 = epoch
    new_t0,new_t1,_,_,Wf,Wb = dataset.calculate_timestep2(
            target_dw,t0)

    # support "state change" glitches, like J2021+4016
    if f1_only:
        f0_min = f0_max = 0


    tbackward = t0-new_t0
    tforward = new_t1-t0
    min_time = max(min_time,max(tbackward,tforward))
    print('dt=%.2f'%min_time)
    prec = model.get_typical_F0_F1(min_time)[-1]
    if (prec > prec_tol):
        # timing noise will affect our pre-glitch solution too much
        # so that an F0/F1 fit will be insufficiently good
        print('problem')
        return

    # save original dataset properties
    old_t0,old_t1 = dataset._t0,dataset._t1
    old_mjds = dataset.get_mjds()
    old_tdbs = dataset.get_tdbs()

    # find t0 corresponding to epoch guess (in TDB)
    a = np.argmin(np.abs(old_tdbs-t0))
    topo_t0 = old_mjds[a]

    ## set new bounds and evaluate the phase on them
    dataset.set_timerange(topo_t0-min_time,topo_t0+min_time)
    phase = model.phase(dataset.get_ts())
    weights = dataset.get_weights()
    new_tdbs = dataset.get_tdbs()
    print(new_tdbs[0],new_tdbs[-1],t0)
    fit_mask = new_tdbs < t0
    fit_phase = phase[fit_mask].astype(float)
    dt_sec = ((new_tdbs-t0)*86400).astype(float)
    p = np.polyfit(dt_sec[fit_mask],fit_phase,2)

    # for comparison later
    Wf_used = np.sum(weights[new_tdbs > t0])
    Wb_used = np.sum(weights[new_tdbs < t0])

    # TMP? sanity check of log likelihood
    logl_init = np.sum(np.log(weights[fit_mask]*dataset._lct(fit_phase%1,use_cache=True)+(1-weights[fit_mask])))
    print('logl_init=%.2f'%logl_init)

    model_phase = np.polyval(p,dt_sec)
    fit_mask = ~fit_mask
    fit_weights = weights[fit_mask]
    fit_dtdbs = new_tdbs[fit_mask] - t0
    fit_phase = model_phase[fit_mask]

    best_z,best_f0,best_f1,best_phase,df0s,df1s = f0_f1_zgrid(
            fit_dtdbs,fit_phase,fit_weights,
            f0_tol=0.10,f1_tol=0.10,
            f0_min=f0_min,f0_max=f0_max,
            f1_min=f1_min,f1_max=f1_max,
            m=4)
    if len(df0s)<=1:
        df0 = 0.
    else:
        df0 = abs(df0s[1]-df0s[0])
    if len(df1s)<=1:
        df1 = 0.
    else:
        df1 = abs(df1s[1]-df1s[0])

    # now try a refinement
    print('Found best-fitting statistic Z2=%.2f'%(best_z))

    if not skip_refine:
        best_z,best_f0_ref,best_f1_ref,best_phase_ref,df0s_ref,df1s_ref = f0_f1_zgrid(
                fit_dtdbs,fit_phase + best_phase,fit_weights,
                f0_tol=0.01,f1_tol=0.01,
                f0_min=0 if f1_only else -df0,f0_max=0 if f1_only else df0,
                f1_min=-df1,f1_max=df1,
                m=10)

        print('Found best-fitting statistic Z2=%.2f'%(best_z))
        best_f0 += best_f0_ref
        best_f1 += best_f1_ref
        best_phase += best_phase_ref
        glitch_phase = fit_phase + best_phase

    # now, determine the phase offset of the post-glitch solution
    if f1_only:
        ph0 = 0
    else:
        dom = np.linspace(0,1,101)
        cod = np.asarray([np.sum(np.log(fit_weights*dataset._lct((glitch_phase+x)%1,use_cache=True)+(1-fit_weights))) for x in dom])
        ph0 = dom[np.argmax(cod)]

    # extrapolate to all phases
    post_phase = ph0 + ((new_tdbs-t0)*86400)*best_f0 + ((new_tdbs-t0)*86400)**2*best_f1
    logl1 = np.cumsum(np.log(weights*dataset._lct(model_phase%1)+(1-weights)))
    logl2 = np.cumsum(np.log(weights*dataset._lct((model_phase+post_phase)%1)+(1-weights))[::-1])[::-1]
    a = np.argmax(logl1+logl2)
    print('Logl so far: %.2f'%(logl1[a]+logl2[a]))
    best_epoch = new_tdbs[a]

    # now, because we have changed the epoch, we need a quick final
    # refinement of the glitch solution
    # if the change is small, just re-optimize.  If it's big, we basically
    # should run the search again, so recurse.
    print('(best_epoch-guess_epoch)=%.3f days'%(best_epoch-t0))
    if abs(best_epoch-t0) > epoch_tol:
        if max_recurse > 0:
            print('New epoch too far from initial guess, recoursing to recursing.')
            dataset.set_timerange(old_t0,old_t1)
            return solve_glitch(model,dataset,best_epoch,prec_tol=prec_tol,
                    target_dw=target_dw,min_time=min_time,
                    max_recurse=max_recurse-1,
                    #f0_min = best_f0-1e-6,f0_max=best_f0+1e-6,
                    #f1_min = best_f1-1e-14,f1_max=best_f1+1e-14,
                    f0_min = f0_min, f0_max=f0_max,
                    f1_min = f1_min, f1_max=f1_max,
                    f1_only=f1_only,epoch_tol=epoch_tol,
                    skip_refine=skip_refine)
        else:
            print('Could not converge on best epoch, sticking with guess.')
            best_epoch = t0

    post_phase = ph0 + ((new_tdbs-best_epoch)*86400)*best_f0 + ((new_tdbs-best_epoch)*86400)**2*best_f1
    fit_mask = new_tdbs >= best_epoch
    fit_dtdbs = new_tdbs[fit_mask] - best_epoch
    fit_phase = model_phase[fit_mask] + post_phase[fit_mask]
    fit_weights = weights[fit_mask]
    old_z = best_z
    best_z,best_f0_ref,best_f1_ref,best_phase_ref,df0s_ref,df1s_ref = f0_f1_zgrid(
            fit_dtdbs,fit_phase,fit_weights,
            f0_tol=0.01,f1_tol=0.01,
            f0_min=0 if f1_only else -2*df0,f0_max=0 if f1_only else 2*df0,
            f1_min=-2*df1,f1_max=2*df1,
            m=10)
    # TODO
    # NB by definition above spacing will search out to 0.1*2 spacing in
    # F0 and F1, seems large enough to handle a modest epoch shift but is
    # also fragile.  Could try an analytic adjustment first, e.g. adjust
    # frequency by the appropriate amt.
    print('Found best-fitting statistic Z2=%.2f'%(best_z))

    # now, re-determine the phase offset of the post-glitch solution
    best_f0 += best_f0_ref
    best_f1 += best_f1_ref
    glitch_phase = model_phase[fit_mask] + (fit_dtdbs*86400)*best_f0 + (fit_dtdbs*86400)**2*best_f1
    if f1_only:
        ph0 = 0
        logl_final = np.sum(np.log(fit_weights*dataset._lct((glitch_phase)%1,use_cache=True)+(1-fit_weights)))
    else:
        dom = np.linspace(0,1,101)
        cod = np.asarray([np.sum(np.log(fit_weights*dataset._lct((glitch_phase+x)%1,use_cache=True)+(1-fit_weights))) for x in dom])
        logl_final = cod.max()
        print('logl_final=%.2f'%logl_final)
        ph0 = dom[np.argmax(cod)]
    epoch_phase = ph0 # NB because the glitch portion is 0 by definition and we want to exclude the model phase
    glitch_phase += ph0

    # final check on phase with model parameters
    final_phase = model_phase.copy()
    final_phase[fit_mask] = glitch_phase

    # make phase appropriate for fitting the full model
    new_t0,new_t1 = dataset._t0,dataset._t1
    dataset.set_timerange(old_t0,dataset._t1)
    target_phase = model.phase(dataset.get_ts())
    target_phase[-len(model_phase):] = final_phase
    dataset.set_timerange(old_t0,old_t1)

    # OK, so at this point, we would want to add the glitch to the model,
    # then because that presumably extrapolates somewhat, extend the data
    # set past that and re-do the fit (with glitch parameters free -- would
    # definitely require a new model)

    # perhaps writing out as a temporary file and then re-loading is the
    # best bet, but I dunno.  Will leave that for tomorrow!
    glitch_p = [best_epoch,epoch_phase,best_f0,best_f1*2]
    return glitch_p,target_phase,[new_t0,new_t1],logl_init,logl_final,locals()

def finish_glitch(model,dataset,solve_glitch_results,p_tn,logl_thresh=1.5,
        ngrid=1024):
    """ Copy the model, extend it to the post-glitch period, and re-fit it
    using the target phase.

    Parameters
    ----------
    model : TMTNJoint
        The object used to furnish solve_glitch_results.
    dataset : DataSet
        Accompanying DataSet object
    solve_glitch_results
        The entire return values of solve_glitch!
    p_tn : tuple
        Timing noise parameters
    logl_thresh : float
        The ratio of the total log likelihood to the pre-glitch likelihood.
        For an ideal solution with uniform data this number should be
        close to 2.0.  If too low, it indicates a poor solution.
    """
    model = deepcopy(model)
    glitch_p = solve_glitch_results[0]
    utils.add_glitch_component(glitch_p,model.tm().model)
    target_phase = solve_glitch_results[1]
    old_t0,old_t1 = dataset._t0,dataset._t1
    new_t1 = solve_glitch_results[2][1]
    dataset.set_timerange(old_t0,new_t1)

    stride = len(target_phase)//ngrid
    subset_mask = np.zeros(len(target_phase),dtype=bool)
    subset_mask[::stride] = True

    nharm = len(model.tn().freqs)
    dataset.apply_toa_mask(dataset.expand_mask(subset_mask))
    tm_tweak = model.tm().get_fixed(deepcopy(dataset.get_ts()))
    tn_tweak = dataset.get_tn(nharm,
            grid_mjds=dataset.get_tdbs(),add_tobs=0)
    mtweak = common.TMTNJoint(tm_tweak,tn_tweak)
    good_fit = common.fit_model(mtweak,target_phase[subset_mask],
            eff_wn=1e-2,H_tn=mtweak.get_htn(p_tn))
    if not good_fit:
        print('Bad gaussian fit!')
    # clear subset mask
    dataset.set_timerange(old_t0,new_t1)

    # NB explicit copy here to avoid chaos when we change the timerange
    mnew = mtweak.change_timerange(deepcopy(dataset.get_ts()))
    h_tn = mnew.get_htn(p_tn)
    LF = common.PhotonLikelihoodFit(mnew,dataset,H_tn=h_tn)
    results_tn = LF.fit(H_tn=h_tn,fit_kwargs=dict(gtol=1e-3,maxiter=200))
    return mnew,locals()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Solve for a glitch.")
    parser.add_argument("jname",help="pulsar name (e.g. J0101-6422)")
    parser.add_argument("--from_extend",help="Use the latest ephemeris from extend_ephemeris and get glitch epoch estimate from it.",action='store_true',default=False)
    parser.add_argument("--epoch",
            help="Guess for glitch epoch (MJD).  Must be provided in absence of --freom_extend.",type=float,default=None)

    args = parser.parse_args()
    jname = args.jname.rstrip('/')

    timingdir = '/tank/kerrm/fermi_data/pulsar_pipeline/%s/timing'%jname
    if args.from_extend:
        extend_pars = glob.glob(timingdir + '/%s_extend_*.par'%(jname))
        if len(extend_pars) == 0:
            raise ValueError('No output from extend_ephemeris.')
            # TODO -- potentially add a sanity check that extend FINISH
            # is actually later than the best TNFIT, but meh
        parfile = sorted(extend_pars)[-1] 
        print('Using ephemeris %s from extend_ephemeris routine.'%(
            parfile))
        # TODO -- get glitch
        epoch = float(utils.get_par_value(parfile,'LATGLITCH'))
    else:
        if args.epoch is None:
            raise ValueError('Must provide glitch guess if not using the output of extend_ephemeris.')
        epoch = args.epoch
        parfile = timingdir + '/%s_tnfit.par'%(jname)

    d,model,(popt,popte) = common.load_data_and_model(jname,
            use_tnfit=True,override_par=parfile,clobber=False)
    stuff = solve_glitch(model,d,epoch)
    mnew, local_vals = finish_glitch(model,d,stuff,popt)
    with open(timingdir + '/%s_glitch.par'%(jname),'w') as f:
        f.write(common.as_tnfit_parfile(mnew,d,popt,p_tne=popte))


    # TODO -- some diagnostic plots from glitch fit
# TODO diagnostic of whether we have good likelihood following the glitch
# fit.  Will we ever actually automate this?
# TODO implement binned likelihoods
# TODO implement pick up from extend ephemeris

